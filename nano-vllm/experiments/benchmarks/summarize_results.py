from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "offline_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL results.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--markdown-path", default=None)
    filtering = parser.add_mutually_exclusive_group()
    filtering.add_argument("--include-errors", "--include-failures", action="store_true",
                           help="Compatibility flag; errors are included by default.")
    filtering.add_argument("--only-success", action="store_true", help="Explicitly omit failed runs.")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path).expanduser()
    if not result_path.is_file():
        raise FileNotFoundError(f"Results file not found: {result_path}")
    records = []
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Results file is empty: {result_path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in: {result_path}")
    return records


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def config_value(record: dict[str, Any], key: str) -> Any:
    return record.get("config", {}).get(key)


def make_markdown(
    records: list[dict[str, Any]], include_failures: bool = True, *, only_success: bool = False,
) -> str:
    rows = []
    for record in records:
        if (record.get("error") or record.get("cleanup_error")) and (only_success or not include_failures):
            continue
        rows.append(
            [
                record.get("timestamp"),
                record.get("engine"),
                Path(str(record.get("model", ""))).name,
                record.get("num_requests"),
                config_value(record, "input_len"),
                config_value(record, "output_len"),
                record.get("input_tokens"),
                record.get("output_tokens"),
                record.get("elapsed_s"),
                record.get("output_tokens_per_s"),
                record.get("total_tokens_per_s"),
                record.get("peak_memory_gb"),
                record.get("oom"),
                record.get("failure_stage", ""),
                record.get("error", ""),
                record.get("cleanup_error", ""),
            ]
        )

    headers = [
        "timestamp",
        "engine",
        "model",
        "requests",
        "input_len",
        "output_len",
        "input_tokens",
        "output_tokens",
        "elapsed_s",
        "out_tok/s",
        "total_tok/s",
        "peak_caller_GiB",
        "oom",
        "failure_stage",
        "error",
        "cleanup_error",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            fmt(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
            for value in row
        ) + " |")
    if not rows:
        lines.append("\n没有匹配记录（可能已被 --only-success 过滤）。")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    try:
        records = load_records(args.results_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot summarize results: {exc}") from exc
    markdown = make_markdown(records, only_success=args.only_success)
    if args.markdown_path:
        path = Path(args.markdown_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
