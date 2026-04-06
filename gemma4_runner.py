#!/usr/bin/env python3
"""
Run Gemma 4 26B A4B chat inference.

Supports:
- Hugging Face model ID (downloads/cached model)
- Local pre-downloaded model folder via --model-path
- Dry-run mode that validates arguments without loading the model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_PROMPT = "Write a short joke about saving RAM."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 26B A4B for a sample/user prompt."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Path to a local downloaded model directory (preferred for large model setups).",
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
        help="Sampling temperature. Set to 0 for greedy decoding.",
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
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more GPU memory).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not try downloading files from remote hubs.",
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
        # Basic check for a Hugging Face model folder structure.
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing config.json in model folder: {model_dir}"
            )
        return str(model_dir), True
    return args.model_id, bool(args.local_files_only)


def run_inference(args: argparse.Namespace, model_source: str, local_only: bool) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[info] CUDA GPU detected: {gpu_name}")
    else:
        print("[warn] CUDA GPU not detected. This model is intended for GPU inference.")

    use_4bit = not args.no_4bit
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"[info] Loading model from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()

    messages = [
        {"role": "user", "content": args.prompt},
    ]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Loaded tokenizer does not support chat templates. "
            "Please upgrade transformers and try again."
        )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded.strip()


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
        print(f"[ok] 4-bit quantization: {not args.no_4bit}")
        print(f"[ok] Prompt: {args.prompt}")
        return 0

    try:
        answer = run_inference(args, model_source, local_only)
    except Exception as exc:
        print(f"[error] Inference failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Model Output ===")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
