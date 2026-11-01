#!/usr/bin/env python3
"""Generate one offline, pinned MLX proposal for the local edit-format task.

The proposer receives only the supplied prompt. Candidate evaluation and
acceptance are handled by the separate black-box evaluator and benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True,
                        help="pinned snapshot commit; local snapshot directory must end in this value")
    parser.add_argument("--model-path", required=True, type=Path,
                        help="local MLX model snapshot directory; network downloads are disabled")
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve(strict=True)
    prompt_path = args.prompt.resolve(strict=True)
    if not model_path.is_dir() or model_path.name != args.revision:
        raise SystemExit("model path must be the local snapshot directory named by --revision")
    if not 0.0 <= args.temperature <= 2.0 or not 0.0 < args.top_p <= 1.0:
        raise SystemExit("temperature/top-p outside supported bounds")
    if not 1 <= args.max_tokens <= 8192:
        raise SystemExit("max-tokens must be in [1, 8192]")
    if not args.attempt_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.attempt_id):
        raise SystemExit("attempt-id must contain only letters, digits, underscore, and hyphen")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    prompt_sha = sha256_bytes(prompt_bytes)
    offline_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    os.environ.update(offline_env)

    started = time.perf_counter()
    metadata: dict[str, object] = {
        "schema": "vare.mlsys_local_proposal_generation.v1",
        "attempt_id": args.attempt_id,
        "status": "loading",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "prompt_sha256": prompt_sha,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "network_downloads": "disabled",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    atomic_write(args.output_dir / "metadata.json", (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())

    partial_path = args.output_dir / "response.partial.txt"
    try:
        # Import only after offline mode is set, and only inside this process.
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.set_default_device(mx.gpu)
        mx.random.seed(args.seed)
        metadata["mlx_version"] = importlib.metadata.version("mlx")
        metadata["mlx_lm_version"] = importlib.metadata.version("mlx-lm")
        load_started = time.perf_counter()
        model, tokenizer = load(str(model_path))
        load_seconds = time.perf_counter() - load_started
        metadata["status"] = "loaded"
        metadata["load_seconds"] = round(load_seconds, 6)
        atomic_write(args.output_dir / "metadata.json",
                     (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
        messages = [
            {"role": "system", "content": "Propose concise, behavior-preserving Python code changes. Do not claim tests passed."},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(chat_prompt))
        sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
        generation_started = time.perf_counter()
        completion_tokens = 0
        last = None
        with partial_path.open("w", encoding="utf-8") as output:
            for result in stream_generate(model, tokenizer, chat_prompt, max_tokens=args.max_tokens,
                                          sampler=sampler):
                output.write(result.text)
                completion_tokens += 1
                last = result
                if completion_tokens % 32 == 0:
                    output.flush()
                    os.fsync(output.fileno())
            output.flush()
            os.fsync(output.fileno())
        generation_seconds = time.perf_counter() - generation_started
        response = partial_path.read_bytes()
        final_path = args.output_dir / "response.txt"
        partial_path.replace(final_path)
        metadata.update({
            "status": "complete",
            "response_sha256": sha256_bytes(response),
            "response_bytes": len(response),
            "prompt_tokens": getattr(last, "prompt_tokens", prompt_tokens),
            "completion_tokens": getattr(last, "generation_tokens", completion_tokens),
            "generation_tokens_per_second": getattr(last, "generation_tps", None),
            "finish_reason": getattr(last, "finish_reason", None),
            "load_seconds": round(load_seconds, 6),
            "generation_seconds": round(generation_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started, 6),
            "active_memory_bytes": mx.get_active_memory(),
            "peak_memory_bytes": mx.get_peak_memory(),
        })
        atomic_write(args.output_dir / "metadata.json",
                     (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    except BaseException as exc:
        metadata.update({
            "status": "failed",
            "failure_type": type(exc).__name__,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "partial_response_sha256": sha256_bytes(partial_path.read_bytes()) if partial_path.exists() else None,
        })
        atomic_write(args.output_dir / "metadata.json",
                     (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
        raise
    print(json.dumps({"attempt_id": args.attempt_id, "status": "complete",
                      "response_sha256": metadata["response_sha256"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
