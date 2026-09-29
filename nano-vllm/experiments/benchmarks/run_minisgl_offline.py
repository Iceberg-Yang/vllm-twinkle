from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    append_jsonl,
    build_failure_record,
    build_success_record,
    config_from_args,
    make_random_token_prompts,
    print_summary,
    reset_cuda_stats,
    synchronize_cuda,
    validate_config,
)


ENGINE = "mini-SGLang"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline mini-SGLang benchmark.")
    add_common_args(parser, ENGINE)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--cache-type", choices=["radix", "naive"], default="radix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args, ENGINE)
    stage = "validate_config"
    llm = None

    try:
        validate_config(config)
        if args.page_size <= 0 or args.cuda_graph_max_bs <= 0:
            raise ValueError("--page-size and --cuda-graph-max-bs must be positive")
        stage = "init"
        import torch
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM
        from transformers import AutoTokenizer

        stage = "load_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        prompts = make_random_token_prompts(
            config.num_requests,
            config.input_len,
            len(tokenizer),
            config.seed,
        )

        dtype = getattr(torch, config.dtype) if config.dtype else torch.bfloat16
        llm_kwargs = {
            "dtype": dtype,
            "max_seq_len_override": config.max_model_len,
            "page_size": args.page_size,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "cache_type": args.cache_type,
        }
        if config.max_num_batched_tokens is not None:
            llm_kwargs["max_extend_tokens"] = config.max_num_batched_tokens
        if config.max_num_seqs is not None:
            llm_kwargs["max_running_req"] = config.max_num_seqs
        if config.gpu_memory_utilization is not None:
            llm_kwargs["memory_ratio"] = config.gpu_memory_utilization

        stage = "load_model"
        llm = LLM(config.model, **llm_kwargs)
        sampling_params = SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=config.output_len,
        )

        stage = "warmup"
        warmup_prompts = make_random_token_prompts(
            config.num_requests, config.input_len, len(tokenizer), config.seed + 1,
        )
        llm.generate(
            warmup_prompts,
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
        )
        synchronize_cuda()

        stage = "benchmark"
        reset_cuda_stats()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        synchronize_cuda()
        elapsed_s = time.perf_counter() - start

        input_tokens = sum(len(prompt) for prompt in prompts)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        record = build_success_record(
            config=config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed_s,
            extra={
                "runner": __file__,
                "effective_dtype": str(dtype),
                "page_size": args.page_size,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "cache_type": args.cache_type,
            },
        )
    except Exception as exc:
        record = build_failure_record(config, exc, stage)
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception as exc:
                record["cleanup_error"] = repr(exc)
                if not record.get("error"):
                    record["error"] = repr(exc)
                    record["failure_stage"] = "cleanup"

    append_jsonl(record, config.results_path)
    print_summary(record)
    if record.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
