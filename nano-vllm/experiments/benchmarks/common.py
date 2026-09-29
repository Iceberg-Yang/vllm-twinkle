from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "offline_results.jsonl"
DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"


@dataclass
class OfflineBenchmarkConfig:
    engine: str
    model: str
    num_requests: int
    input_len: int
    output_len: int
    seed: int
    max_model_len: int
    max_num_batched_tokens: int | None
    max_num_seqs: int | None
    gpu_memory_utilization: float | None
    enforce_eager: bool | None
    dtype: str | None
    results_path: str
    notes: str


def add_common_args(parser: argparse.ArgumentParser, engine: str) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Local model path.")
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--notes", default="")
    if engine == "nano-vLLM":
        parser.add_argument("--enforce-eager", action="store_true")


def config_from_args(args: argparse.Namespace, engine: str) -> OfflineBenchmarkConfig:
    return OfflineBenchmarkConfig(
        engine=engine,
        model=os.path.expanduser(args.model),
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=getattr(args, "enforce_eager", None),
        dtype=args.dtype,
        results_path=args.results_path,
        notes=args.notes,
    )


def validate_config(config: OfflineBenchmarkConfig) -> None:
    for name in ("num_requests", "input_len", "output_len", "max_model_len",
                 "max_num_batched_tokens", "max_num_seqs"):
        value = getattr(config, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    ratio = config.gpu_memory_utilization
    if ratio is not None and (not math.isfinite(ratio) or not 0 < ratio <= 1):
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if config.input_len + config.output_len > config.max_model_len:
        raise ValueError("input_len + output_len must not exceed max_model_len")
    if config.input_len + 2 > config.max_model_len:
        raise ValueError("max_model_len must also allow two warmup decode tokens")
    if config.engine == "Transformers":
        for name in ("max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization"):
            if getattr(config, name) is not None:
                raise ValueError(f"Transformers does not support --{name.replace('_', '-')}")


def add_nanovllm_source_to_path() -> None:
    """Compatibility entry point: require installed nano-vLLM; never alter sys.path."""
    if importlib.util.find_spec("nanovllm") is None:
        raise ImportError(
            "nanovllm is not installed in this Python environment. "
            "Install the intended nano-vLLM version in its isolated environment first; "
            "this benchmark does not bundle nano-vLLM sources."
        )


def resolve_nanovllm_dtype(model_dtype: Any, requested_dtype: str | None) -> str:
    """nano-vLLM uses hf_config.dtype directly and cannot override it here."""
    aliases = {"half": "float16", "float": "float32", "fp16": "float16",
               "bf16": "bfloat16", "fp32": "float32"}

    def normalize(value: Any) -> str:
        name = str(value).removeprefix("torch.").lower()
        return aliases.get(name, name)

    effective = normalize(model_dtype)
    if effective not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported or missing model config dtype: {model_dtype!r}")
    if requested_dtype not in (None, "auto") and normalize(requested_dtype) != effective:
        raise ValueError(
            f"nano-vLLM cannot override model dtype {effective} with {requested_dtype!r}; "
            "ModelRunner uses torch.set_default_dtype(hf_config.dtype)"
        )
    return effective


def make_random_token_prompts(
    num_requests: int,
    input_len: int,
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    upper = max(1, vocab_size - 1)
    return [[rng.randint(0, upper) for _ in range(input_len)] for _ in range(num_requests)]


def reset_cuda_stats() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def synchronize_cuda() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_gb() -> float | None:
    """Legacy name: GiB allocated by caller-process torch on its current CUDA device.

    Excludes other processes, allocator reservations and non-torch allocations.
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def torch_env() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": repr(exc)}
    info: dict[str, Any] = {
        "torch": getattr(torch, "__version__", None),
        "cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def installed_versions() -> dict[str, str]:
    """Read local distribution metadata only; no framework imports or network."""
    versions = {}
    for name in ("torch", "transformers", "accelerate", "nanovllm", "nano-vllm",
                 "minisgl", "mini-sglang", "vllm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def base_record(config: OfflineBenchmarkConfig) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "env": {**torch_env(), "python": platform.python_version(),
                "installed_versions": installed_versions()},
        "memory_unit": "GiB",
        "memory_scope": "caller_process_torch_allocator",
    }


def build_success_record(
    config: OfflineBenchmarkConfig,
    input_tokens: int,
    output_tokens: int,
    elapsed_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = base_record(config)
    total_tokens = input_tokens + output_tokens
    record.update(
        {
            "engine": config.engine,
            "model": config.model,
            "num_requests": config.num_requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": elapsed_s,
            "total_tokens_per_s": total_tokens / elapsed_s if elapsed_s > 0 else None,
            "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else None,
            "peak_memory_gb": peak_memory_gb(),
            "oom": False,
            "error": None,
        }
    )
    if extra:
        record.update(extra)
    return record


def build_failure_record(
    config: OfflineBenchmarkConfig,
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    record = base_record(config)
    record.update(
        {
            "engine": config.engine,
            "model": config.model,
            "num_requests": config.num_requests,
            "input_tokens": None,
            "output_tokens": None,
            "elapsed_s": None,
            "total_tokens_per_s": None,
            "output_tokens_per_s": None,
            "peak_memory_gb": peak_memory_gb(),
            "oom": "out of memory" in str(error).lower(),
            "error": repr(error),
            "failure_stage": stage,
        }
    )
    return record


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def print_summary(record: dict[str, Any]) -> None:
    fields = [
        "engine",
        "num_requests",
        "input_tokens",
        "output_tokens",
        "elapsed_s",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "peak_memory_gb",
        "memory_unit",
        "memory_scope",
        "oom",
        "error",
    ]
    for field in fields:
        print(f"{field}: {record.get(field)}")
