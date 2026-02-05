import sys
import re
import torch
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal, Generator, Callable
from tokenizers import Tokenizer
from qwen3 import Qwen3Model, KVCache, QWEN_CONFIG_06_B


def download_file(url: str, out_dir: str=".", backup_url: Optional[str]=None) -> Path:
    """Download *url* into *out_dir* with an optional mirror fallback."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name
    dest = out_path / filename

    def _download(u: str) -> bool:
        try:
            with urllib.request.urlopen(url=u) as r:
                size_remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True

                block = 1024 * 1024  # 1 MiB
                downloaded = 0
                with open(dest, "wb") as f:
                    while chunk := r.read(block):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / {size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False

    if _download(u=url):
        return dest

    if backup_url:
        print(f"Primary URL ({url}) failed. \nTrying backup URL ({backup_url})...,")
        if _download(u=backup_url):
            return dest

    raise RuntimeError(f"Failed to download {filename} from both mirrors.")


def download_qwen3_small(kind: Literal["base", "reasoning"]="base", tokenizer_only: bool=False, out_dir: str=".") -> None:
    files = {
        "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
        "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(url=primary, out_dir=out_dir, backup_url=backup)


class Qwen3Tokenizer:

    _SPECIALS: List[str] = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(pattern=r"(<\|[^>]+?\|>)")

    apply_chat_template: bool
    add_generation_prompt: bool
    add_thinking: bool

    _tok: Tokenizer
    _special_to_id: Dict[str, Optional[int]]

    pad_token: str
    pad_token_id: Optional[int]
    eos_token: str
    eos_token_id: Optional[int]

    def __init__(self, tokenizer_file_path: str | Path ="tokenizer.json",
                 apply_chat_template: bool=False,
                 add_generation_prompt: bool=False,
                 add_thinking: bool=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_path}' not found. ")

        self._tok: Tokenizer = Tokenizer.from_file(path=str(tok_path))
        self._special_to_id: Dict[str, Optional[int]] = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt: str, chat_wrapped: bool | None = None) -> list[int]:
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if "\n" not in stripped:
            tid0 = self._special_to_id.get(stripped)
            if tid0 is not None:
                return [tid0]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            tid = self._special_to_id.get(part)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(ids=token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg: str) -> str:
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def load_model_and_tokenizer(which_model: Literal["base", "reasoning"], device: torch.device,
                             use_compile: bool, local_dir: str="qwen3") -> Tuple[Qwen3Model, Qwen3Tokenizer]:
    if which_model == "base":

        download_qwen3_small(kind="base", tokenizer_only=False, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":

        download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer

