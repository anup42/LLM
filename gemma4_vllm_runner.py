#!/usr/bin/env python3
"""
Run Gemma 4 26B A4B using vLLM.

Supports:
- Hugging Face model ID
- Local pre-downloaded model folder via --model-path
- Dry-run mode that validates configuration without loading the model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_PROMPT = "Write a short joke about saving RAM."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 26B A4B for a sample/user prompt with vLLM."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Path to a local downloaded model directory.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt text (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree (set >1 for multi-GPU).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of each GPU memory that vLLM can use.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length in vLLM.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
        help="Model dtype for vLLM.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not allow downloading from remote hubs.",
    )
    parser.add_argument(
        "--download-dir",
        default="",
        help="Optional Hugging Face cache/download directory for vLLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without loading the model.",
    )
    return parser.parse_args()


def resolve_model_source(args: argparse.Namespace) -> tuple[str, bool]:
    if args.model_path:
        model_dir = Path(args.model_path).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_dir}")
        if not model_dir.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {model_dir}")
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing config.json in model folder: {model_dir}"
            )
        return str(model_dir), True
    return args.model_id, bool(args.local_files_only)


def _extract_text(output_list) -> str:
    if not output_list:
        return ""
    first = output_list[0]

    candidates = getattr(first, "outputs", None)
    if candidates and len(candidates) > 0:
        first_candidate = candidates[0]
        text = getattr(first_candidate, "text", None)
        if isinstance(text, str):
            return text.strip()

    if isinstance(first, dict):
        nested = first.get("outputs")
        if isinstance(nested, list) and nested:
            maybe_text = nested[0].get("text")
            if isinstance(maybe_text, str):
                return maybe_text.strip()

    return str(first).strip()


def run_inference(args: argparse.Namespace, model_source: str, local_only: bool) -> str:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected by PyTorch. vLLM requires a working CUDA setup."
        )

    if local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    llm_kwargs = {
        "model": model_source,
        "tokenizer": model_source,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    print(f"[info] Loading vLLM model from: {model_source}")
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    messages = [{"role": "user", "content": args.prompt}]
    try:
        outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        text = _extract_text(outputs)
        if text:
            return text
    except Exception as chat_exc:
        print(f"[warn] llm.chat path failed, falling back to text-generate: {chat_exc}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True, local_files_only=local_only
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    text = _extract_text(outputs)
    if not text:
        raise RuntimeError("No text returned from vLLM generation.")
    return text


def main() -> int:
    args = parse_args()

    try:
        model_source, local_only = resolve_model_source(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("[ok] Dry run passed.")
        print(f"[ok] Model source: {model_source}")
        print(f"[ok] local_files_only: {local_only}")
        print(f"[ok] tensor_parallel_size: {args.tensor_parallel_size}")
        print(f"[ok] gpu_memory_utilization: {args.gpu_memory_utilization}")
        print(f"[ok] max_model_len: {args.max_model_len}")
        print(f"[ok] Prompt: {args.prompt}")
        return 0

    try:
        answer = run_inference(args, model_source, local_only)
    except Exception as exc:
        print(f"[error] Inference failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Model Output (vLLM) ===")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
