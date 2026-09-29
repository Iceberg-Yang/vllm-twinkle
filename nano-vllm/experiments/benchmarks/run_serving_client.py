from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "serving_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible serving endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-rate", type=float, default=0.0, help="Submission rate per second. 0 submits immediately; excess work queues in the client.")
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Socket per-operation timeout in seconds (单次读写超时), not a strict total request deadline.")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_requests", "concurrency", "prompt_words", "max_tokens", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("request_rate", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not args.url.startswith(("http://", "https://")):
        raise ValueError("--url must use http:// or https://")


def make_prompt(index: int, prompt_words: int) -> str:
    prefix = f"Request {index}: explain LLM inference infrastructure."
    filler = " ".join(["benchmark"] * max(0, prompt_words))
    return f"{prefix}\n{filler}"


def build_payload(args: argparse.Namespace, index: int) -> dict[str, Any]:
    prompt = make_prompt(index, args.prompt_words)
    base = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": not args.no_stream,
    }
    if not args.no_stream:
        base["stream_options"] = {"include_usage": True}
    if args.endpoint == "chat":
        base["messages"] = [{"role": "user", "content": prompt}]
    else:
        base["prompt"] = prompt
    return base


def check_response_error(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON response object")
    if "error" in data and data["error"] is not None:
        raise ValueError(f"Endpoint returned error: {data['error']!r}")


def extract_text_from_chunk(data: dict[str, Any], endpoint: str, stream: bool = True) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if endpoint == "chat":
        message = choice.get("delta" if stream else "message") or {}
        return message.get("content") or ""
    return choice.get("text") or ""


def extract_usage_completion_tokens(data: dict[str, Any]) -> int | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if type(value) is int and value >= 0 else None


def run_one_request(
    args: argparse.Namespace, index: int, submitted_at: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    if submitted_at is None:
        submitted_at = start
    result = {
        "ok": False,
        "request_id": index,
        "queue_s": start - submitted_at,
        "ttft_s": None,
        "tpot_s": None,
        "tpot_unit_source": None,
        "output_units": 0,
        "output_unit_source": None,
        "error": None,
    }
    first_token_s: float | None = None
    chunk_count = 0
    completion_tokens: int | None = None
    finished = False

    try:
        body = json.dumps(build_payload(args, index)).encode("utf-8")
        request = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            content_type = getattr(response, "headers", {}).get("Content-Type", "")
            if args.no_stream or "application/json" in content_type.lower():
                data = json.loads(response.read().decode("utf-8"))
                check_response_error(data)
                if not args.no_stream:
                    raise ValueError("Expected SSE stream but received non-streaming JSON")
                if not data.get("choices"):
                    raise ValueError("Non-streaming response has no choices")
                text = extract_text_from_chunk(data, args.endpoint, stream=False)
                completion_tokens = extract_usage_completion_tokens(data)
                result["output_units"] = completion_tokens if completion_tokens is not None else len(text.split())
                result["output_unit_source"] = "usage.completion_tokens" if completion_tokens is not None else "whitespace_words"
            else:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if data_text == "[DONE]":
                        finished = True
                        break
                    data = json.loads(data_text)
                    check_response_error(data)
                    if any(choice.get("finish_reason") is not None for choice in data.get("choices", [])):
                        finished = True
                    usage_tokens = extract_usage_completion_tokens(data)
                    if usage_tokens is not None:
                        completion_tokens = usage_tokens
                    text = extract_text_from_chunk(data, args.endpoint)
                    if text:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - start
                        chunk_count += 1
                if not finished:
                    raise ValueError("Truncated SSE stream: EOF without [DONE] or finish_reason")
                result["output_units"] = completion_tokens if completion_tokens is not None else chunk_count
                result["output_unit_source"] = "usage.completion_tokens" if completion_tokens is not None else "stream_chunks"
                result["ttft_s"] = first_token_s
        result["ok"] = True
    except Exception as exc:
        # Include protocol errors and HTTP 200 error payloads, not just transport errors.
        result["error"] = repr(exc)

    end = time.perf_counter()
    result["latency_s"] = end - start
    result["end_to_end_s"] = end - submitted_at
    if result["ok"] and first_token_s is not None and result["output_units"] > 1:
        result["tpot_s"] = (result["latency_s"] - first_token_s) / (result["output_units"] - 1)
        result["tpot_unit_source"] = result["output_unit_source"]
    return result


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[int(rank)]
    weight = rank - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def summarize(args: argparse.Namespace, results: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    successes = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    ttfts = [r["ttft_s"] for r in successes if r["ttft_s"] is not None]
    tpot_sources = sorted({r.get("tpot_unit_source", r["output_unit_source"])
                           for r in successes if r["tpot_s"] is not None})
    tpots = [r["tpot_s"] for r in successes if r["tpot_s"] is not None] if len(tpot_sources) == 1 else []
    latencies = [r["latency_s"] for r in successes]
    sources = sorted({r["output_unit_source"] for r in successes if r["output_unit_source"]})
    units_by_source = {
        source: sum(r["output_units"] for r in successes if r["output_unit_source"] == source)
        for source in sources
    }
    output_units = sum(units_by_source.values()) if len(sources) == 1 else None

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": args.url,
        "endpoint": args.endpoint,
        "model": args.model,
        "stream": not args.no_stream,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "prompt_words": args.prompt_words,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_s": args.timeout,
        "timeout_scope": "socket_per_operation_not_total_deadline",
        "elapsed_s": elapsed_s,
        "completed_count": len(results),
        "success_count": len(successes),
        "error_count": len(errors),
        "success_rate": len(successes) / args.num_requests if args.num_requests > 0 else None,
        "latency_scope": "worker_start_to_response_end_excluding_client_queue",
        "latency_percentile_scope": "successful_requests_only",
        "all_latency_percentile_scope": "all_completed_requests_including_failures",
        "ttft_scope": "worker_start_to_first_nonempty_text_chunk_success_only",
        "tpot_scope": "success_only_estimate_from_usage_or_chunks_not_exact_token_timing",
        "tpot_unit_sources": tpot_sources,
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p95": percentile(ttfts, 0.95),
        "tpot_p50": percentile(tpots, 0.50),
        "tpot_p95": percentile(tpots, 0.95),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "all_latency_p95": percentile([r["latency_s"] for r in results], 0.95),
        "queue_p95": percentile([r.get("queue_s") for r in results], 0.95),
        "end_to_end_p95": percentile([r.get("end_to_end_s") for r in results], 0.95),
        "end_to_end_scope": "submission_to_response_end_all_completed_requests",
        "queue_scope": "client_submission_to_worker_start_all_completed_requests",
        "output_units": output_units,
        "output_units_per_s": output_units / elapsed_s if output_units is not None and elapsed_s > 0 else None,
        "output_unit_sources": sources,
        "output_units_by_source": units_by_source,
        "output_units_per_s_by_source": {
            source: units / elapsed_s if elapsed_s > 0 else None
            for source, units in units_by_source.items()
        },
        "output_throughput_scope": "successful_outputs_over_full_round_elapsed",
        "errors": errors,
        "request_results": sorted(results, key=lambda r: r["request_id"]),
        "error": None,
        "notes": args.notes,
    }


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def print_summary(record: dict[str, Any]) -> None:
    fields = [
        "success_count",
        "error_count",
        "success_rate",
        "elapsed_s",
        "ttft_p50",
        "ttft_p95",
        "tpot_p50",
        "tpot_p95",
        "tpot_unit_sources",
        "tpot_scope",
        "latency_p50",
        "latency_p95",
        "latency_percentile_scope",
        "all_latency_p95",
        "queue_p95",
        "end_to_end_p95",
        "output_units",
        "output_units_per_s",
        "output_unit_sources",
        "output_units_per_s_by_source",
        "error",
    ]
    for field in fields:
        print(f"{field}: {record.get(field)}")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    results: list[dict[str, Any]] = []

    error = None
    stage = "validate_args"
    try:
        validate_args(args)
        stage = "requests"
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for index in range(args.num_requests):
                submitted_at = time.perf_counter()
                futures.append(executor.submit(run_one_request, args, index, submitted_at))
                if args.request_rate > 0 and index + 1 < args.num_requests:
                    time.sleep(1.0 / args.request_rate)
            for future in as_completed(futures):
                results.append(future.result())
    except Exception as exc:
        error = repr(exc)

    elapsed_s = time.perf_counter() - start
    record = summarize(args, results, elapsed_s)
    if error:
        record.update(error=error, failure_stage=stage)
    append_jsonl(record, args.results_path)
    print_summary(record)
    if error or record["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
