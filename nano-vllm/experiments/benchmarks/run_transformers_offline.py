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


ENGINE = "Transformers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Transformers benchmark.")
    add_common_args(parser, ENGINE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args, ENGINE)
    stage = "validate_config"

    try:
        validate_config(config)
        stage = "init"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        stage = "load_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        prompts = make_random_token_prompts(
            config.num_requests,
            config.input_len,
            len(tokenizer),
            config.seed,
        )

        stage = "load_model"
        # An explicit device map requires accelerate; never spread across GPUs.
        model_kwargs = {"device_map": {"": "cuda:0"}}
        if config.dtype:
            model_kwargs["torch_dtype"] = getattr(torch, config.dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        model.eval()

        input_ids = torch.tensor(prompts, dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)

        stage = "warmup"
        warmup_prompts = make_random_token_prompts(
            config.num_requests, config.input_len, len(tokenizer), config.seed + 1,
        )
        warmup_ids = torch.tensor(warmup_prompts, dtype=torch.long, device=model.device)
        with torch.inference_mode():
            model.generate(
                input_ids=warmup_ids,
                attention_mask=torch.ones_like(warmup_ids),
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=True,
                temperature=0.6,
                top_k=0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
                forced_eos_token_id=None,
            )
        synchronize_cuda()
        del warmup_ids

        stage = "benchmark"
        reset_cuda_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.output_len,
                min_new_tokens=config.output_len,
                do_sample=True,
                temperature=0.6,
                top_k=0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
                forced_eos_token_id=None,
            )
        synchronize_cuda()
        elapsed_s = time.perf_counter() - start

        new_token_ids = output_ids[:, input_ids.shape[1] :]
        input_tokens = int(input_ids.numel())
        output_tokens = int(new_token_ids.numel())
        record = build_success_record(
            config=config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed_s,
            extra={"runner": __file__, "effective_dtype": str(model.dtype),
                   "device_map": {"": "cuda:0"}},
        )
    except Exception as exc:
        record = build_failure_record(config, exc, stage)

    append_jsonl(record, config.results_path)
    print_summary(record)
    if record.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
