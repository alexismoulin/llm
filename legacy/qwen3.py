import torch
import torch.nn as nn
from typing import List, Tuple, Optional, TypedDict

# -------- Types --------

KVPair = Tuple[torch.Tensor, torch.Tensor]

class QwenConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    head_dim: Optional[int]
    qk_norm: bool
    n_kv_groups: int
    rope_base: float
    dtype: torch.dtype

# -------- Config --------

QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Context length that was used to train the model
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of the intermediate dimension in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries and keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory usage
}

# -------- Modules --------

class RMSNorm(nn.Module):
    eps: float
    qwen3_compatible: bool
    scale: nn.Parameter
    shift: Optional[nn.Parameter]

    def __init__(self, emb_dim: int, eps: float=1e-6, bias: bool=False, qwen3_compatible: bool=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class FeedForward(nn.Module):
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1: torch.Tensor = self.fc1(x)
        x_fc2: torch.Tensor = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

# -------- RoPE utilities --------

def compute_rope_params(head_dim: int, theta_base: float=10_000.0, context_length: int=4096,
                        dtype: torch.dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(start=0, end=head_dim, step=2, dtype=dtype)[: (head_dim // 2)].float() / head_dim
    ))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int=0) -> torch.Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes, shape: (1, 1, seq_len, head_dim)
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    num_heads: int
    num_kv_groups: int
    group_size: int
    head_dim: int
    d_out: int

    W_query: nn.Linear
    W_key: nn.Linear
    W_value: nn.Linear
    out_proj: nn.Linear

    q_norm: Optional[RMSNorm]
    k_norm: Optional[RMSNorm]

    def __init__(self, d_in: int, num_heads: int, num_kv_groups: int, head_dim: Optional[int] = None,
                 qk_norm: bool = False, dtype: Optional[torch.dtype] = None):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(
            d_in, self.d_out, bias=False, dtype=dtype
        )
        self.W_key = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0,
                cache: Optional[KVPair] = None) -> Tuple[torch.Tensor, KVPair]:

        b, num_tokens, _ = x.shape

        queries: torch.Tensor = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys: torch.Tensor = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values: torch.Tensor = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat(tensors=[prev_k, keys_new], dim=2)
            values = torch.cat(tensors=[prev_v, values_new], dim=2)
        else:
            start_pos = 0  # reset RoPE
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(repeats=self.group_size, dim=1)
        values = values.repeat_interleave(repeats=self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


class TransformerBlock(nn.Module):
    att: GroupedQueryAttention
    ff: FeedForward
    norm1: RMSNorm
    norm2: RMSNorm

    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int=0,
                cache: Optional[KVPair]=None) -> Tuple[torch.Tensor, KVPair]:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos,cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache
    

class KVCache:
    cache: List[Optional[KVPair]]

    def __init__(self, n_layers: int) -> None:
        self.cache = [None] * n_layers

    def get(self, layer_idx: int) -> Optional[KVPair]:
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: Optional[KVPair]) -> None:
        self.cache[layer_idx] = value

    def get_all(self) -> List[Optional[KVPair]]:
        return self.cache

    def reset(self) -> None:
        for i in range(len(self.cache)):
            self.cache[i] = None


class Qwen3Model(nn.Module):
    tok_emb: nn.Embedding
    trf_blocks: nn.ModuleList
    final_norm: RMSNorm
    out_head: nn.Linear

    cos: torch.Tensor
    sin: torch.Tensor

    cfg: QwenConfig
    current_pos: int

    def __init__(self, cfg: QwenConfig):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim: int = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim: int = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer(name="cos", tensor=cos, persistent=False)
        self.register_buffer(name="sin", tensor=sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache

    def forward(self, in_idx: torch.Tensor, cache: Optional[KVCache]=None) -> torch.Tensor:
        # Forward pass
        tok_embeds: torch.Tensor = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                input=torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                input=torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                cache.update(layer_idx=i, value=new_blk_cache)

        x = self.final_norm(x)
        logits: torch.Tensor = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self) -> None:
        self.current_pos = 0