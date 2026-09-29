from __future__ import annotations

import argparse
import time

from common import (
    add_common_args,
    add_nanovllm_source_to_path,
    append_jsonl,
    build_failure_record,
    build_success_record,
    config_from_args,
    make_random_token_prompts,
    print_summary,
    reset_cuda_stats,
    resolve_nanovllm_dtype,
    synchronize_cuda,
    validate_config,
)


ENGINE = "nano-vLLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline nano-vLLM benchmark.")
    add_common_args(parser, ENGINE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args, ENGINE)
    stage = "validate_config"
    effective_dtype = None

    try:
        validate_config(config)
        stage = "init"
        add_nanovllm_source_to_path()
        from nanovllm import LLM, SamplingParams
        from transformers import AutoConfig, AutoTokenizer

        stage = "validate_dtype"
        hf_config = AutoConfig.from_pretrained(config.model)
        effective_dtype = resolve_nanovllm_dtype(hf_config.dtype, config.dtype)
        stage = "load_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        prompts = make_random_token_prompts(
            config.num_requests,
            config.input_len,
            len(tokenizer),
            config.seed,
        )
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=config.output_len)
            for _ in range(config.num_requests)
        ]

        llm_kwargs = {
            "enforce_eager": bool(config.enforce_eager),
            "max_model_len": config.max_model_len,
        }
        if config.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
        if config.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = config.max_num_seqs
        if config.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization

        stage = "load_model"
        llm = LLM(config.model, **llm_kwargs)

        stage = "warmup"
        warmup_prompts = make_random_token_prompts(
            config.num_requests, config.input_len, len(tokenizer), config.seed + 1,
        )
        llm.generate(
            warmup_prompts,
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2),
            use_tqdm=False,
        )
        synchronize_cuda()

        stage = "benchmark"
        reset_cuda_stats()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        synchronize_cuda()
        elapsed_s = time.perf_counter() - start

        input_tokens = sum(len(prompt) for prompt in prompts)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        record = build_success_record(
            config=config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed_s,
            extra={"runner": __file__},
        )
    except Exception as exc:
        record = build_failure_record(config, exc, stage)

    record["effective_dtype"] = effective_dtype
    append_jsonl(record, config.results_path)
    print_summary(record)
    if record.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
