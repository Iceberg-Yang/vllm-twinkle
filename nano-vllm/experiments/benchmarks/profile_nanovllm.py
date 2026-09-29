from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    add_nanovllm_source_to_path,
    make_random_token_prompts,
    peak_memory_gb,
    reset_cuda_stats,
    synchronize_cuda,
)


DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "results" / "profiles"
DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile nano-vLLM with torch.profiler.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_nanovllm_source_to_path()

    import torch
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = str(Path(args.model).expanduser())
    trace_dir = Path(args.trace_dir).expanduser()
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("nanovllm_%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"{run_name}.json"
    table_path = trace_dir / f"{run_name}_table.txt"
    record_path = trace_dir / f"{run_name}_record.json"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompts = make_random_token_prompts(
        args.num_requests,
        args.input_len,
        len(tokenizer),
        args.seed,
    )
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
        for _ in range(args.num_requests)
    ]

    llm_kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    llm = LLM(model_path, **llm_kwargs)
    llm.generate([[1, 2, 3]], SamplingParams(max_tokens=1), use_tqdm=False)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    reset_cuda_stats()
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        synchronize_cuda()
    elapsed_s = time.perf_counter() - start

    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=80,
    )
    table_path.write_text(table, encoding="utf-8")

    input_tokens = sum(len(prompt) for prompt in prompts)
    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "nano-vLLM",
        "model": model_path,
        "num_requests": args.num_requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else None,
        "peak_memory_gb": peak_memory_gb(),
        "memory_unit": "GiB",
        "memory_scope": "caller_process_torch_allocator",
        "trace_path": str(trace_path),
        "table_path": str(table_path),
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "with_stack": args.with_stack,
        "notes": args.notes,
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"trace_path: {trace_path}")
    print(f"table_path: {table_path}")
    print(f"record_path: {record_path}")
    print(f"elapsed_s: {elapsed_s:.3f}")
    print(f"output_tokens_per_s: {record['output_tokens_per_s']:.3f}")
    print(f"peak_memory_gb: {record['peak_memory_gb']}")


if __name__ == "__main__":
    main()
