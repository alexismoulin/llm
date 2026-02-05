import torch
import re
from typing import Generator, Optional, Literal
from utils import KVCache, Qwen3Model


def render_prompt(prompt: str) -> str:
    template = (
        "You are a helpful math assistant.\n"
        "Solve the problem and write the final "
        "result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Problem:\n{prompt}\n\nAnswer:"
    )
    return template


@torch.inference_mode()
def generate_text_basic_stream_cache(model: Qwen3Model, token_ids: torch.Tensor, max_new_tokens: int,
                                     eos_token_id: Optional[int] = None) -> Generator[torch.Tensor, None, None]:
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out: torch.Tensor = model(in_idx=token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(in_idx=next_token, cache=cache)[:, -1]


Fallback = Literal[
    "number_then_full",  # (default): pick the last simple number, else the whole text
    "number_only",  # pick the last simple number, else return an empty string "";
    "none"  # extract only boxed content, else return empty string "".
]

RE_NUMBER = re.compile(pattern=r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def get_last_boxed(text: str) -> Optional[str]:
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")

    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx:current_idx - 1]


def extract_final_candidate(text: str, fallback: Fallback = "number_then_full") -> str:
    result = ""

    if text:
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]
            elif fallback == "number_then_full":
                result = text
    return result
