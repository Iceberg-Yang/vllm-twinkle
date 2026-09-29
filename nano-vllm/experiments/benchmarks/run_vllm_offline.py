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


ENGINE = "vLLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline vLLM benchmark.")
    add_common_args(parser, ENGINE)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args, ENGINE)
    stage = "validate_config"

    try:
        validate_config(config)
        stage = "init"
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        stage = "load_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        prompts = make_random_token_prompts(
            config.num_requests,
            config.input_len,
            len(tokenizer),
            config.seed,
        )

        llm_kwargs = {
            "model": config.model,
            "max_model_len": config.max_model_len,
        }
        if config.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
        if config.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = config.max_num_seqs
        if config.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
        if config.dtype is not None:
            llm_kwargs["dtype"] = config.dtype
        llm_kwargs["enable_prefix_caching"] = args.enable_prefix_caching

        stage = "load_model"
        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=config.output_len,
            ignore_eos=True,
        )
        vllm_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]

        stage = "warmup"
        warmup_prompts = make_random_token_prompts(
            config.num_requests, config.input_len, len(tokenizer), config.seed + 1,
        )
        llm.generate(
            [{"prompt_token_ids": prompt} for prompt in warmup_prompts],
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
        )
        synchronize_cuda()

        stage = "benchmark"
        reset_cuda_stats()
        start = time.perf_counter()
        outputs = llm.generate(vllm_prompts, sampling_params)
        synchronize_cuda()
        elapsed_s = time.perf_counter() - start

        input_tokens = sum(len(prompt) for prompt in prompts)
        output_tokens = 0
        for output in outputs:
            for completion in output.outputs:
                output_tokens += len(completion.token_ids)

        record = build_success_record(
            config=config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed_s,
            extra={
                "runner": __file__,
                "enable_prefix_caching": args.enable_prefix_caching,
            },
        )
    except Exception as exc:
        record = build_failure_record(config, exc, stage)

    record["enable_prefix_caching"] = args.enable_prefix_caching
    append_jsonl(record, config.results_path)
    print_summary(record)
    if record.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
