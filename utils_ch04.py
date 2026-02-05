import torch
from typing import Optional, Callable, Generator
from utils import KVCache, Qwen3Model, Qwen3Tokenizer
from utils_ch03 import generate_text_basic_stream_cache


def generate_text_stream_concat_flex(model: Qwen3Model, tokenizer: Qwen3Tokenizer, prompt: str, device: torch.device,
                                     max_new_tokens: int, verbose: bool = False,
                                     generate_func: Optional[Callable[..., Generator[torch.Tensor, None, None]]] = None,
                                     **generate_kwargs) -> str:
    if generate_func is None:
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    generated_ids = []
    for token in generate_func(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


def scale_logits_by_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def top_p_filter(probas: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    keep = cumprobas <= top_p
    keep[:, 0] = True

    kept_sorted = torch.where(condition=keep, input=sorted_probas, other=torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(dim=1, index=sorted_idx, src=kept_sorted)
    denom = torch.sum(filtered, dim=1).clamp_min(1e-12)
    return filtered / denom


@torch.inference_mode()
def generate_text_top_p_stream_cache(model: Qwen3Model, token_ids: torch.Tensor, max_new_tokens: int,
                                     eos_token_id: Optional[int] = None, temperature: Optional[float] = 0,
                                     top_p: Optional[float] = None) -> Generator[torch.Tensor, None, None]:
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out: torch.Tensor = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 1.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
            probas = top_p_filter(probas, top_p)
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]
