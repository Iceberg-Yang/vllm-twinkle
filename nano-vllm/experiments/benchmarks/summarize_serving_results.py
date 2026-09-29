from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "serving_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize serving benchmark JSONL results.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--markdown-path", default=None)
    filtering = parser.add_mutually_exclusive_group()
    filtering.add_argument("--include-errors", action="store_true",
                           help="Compatibility flag; errors are included by default.")
    filtering.add_argument("--only-success", action="store_true", help="Explicitly omit failed rounds.")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path).expanduser()
    if not result_path.is_file():
        raise FileNotFoundError(f"Results file not found: {result_path}")
    records = []
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def make_markdown(
    records: list[dict[str, Any]], include_errors: bool = True, *, only_success: bool = False,
) -> str:
    headers = [
        "timestamp",
        "endpoint",
        "model",
        "stream",
        "requests",
        "concurrency",
        "success",
        "errors",
        "success_rate",
        "ttft_p50_s (success-only)",
        "ttft_p95_s (success-only)",
        "tpot_p50_s/unit (success-only estimate)",
        "tpot_p95_s/unit (success-only estimate)",
        "tpot_unit_source",
        "lat_p50_s (success-only)",
        "lat_p95_s (success-only)",
        "lat_p95_s (all)",
        "queue_p95_s (all)",
        "end_to_end_p95_s (all)",
        "out_units/s (success outputs)",
        "unit_source",
        "out_units/s_by_source",
        "error_details",
        "notes",
    ]
    rows = []
    for record in records:
        if (record.get("error_count", 0) or record.get("error") or record.get("errors")) and (only_success or not include_errors):
            continue
        rows.append(
            [
                record.get("timestamp"),
                record.get("endpoint"),
                Path(str(record.get("model", ""))).name,
                record.get("stream"),
                record.get("num_requests"),
                record.get("concurrency"),
                record.get("success_count"),
                record.get("error_count"),
                record.get("success_rate"),
                record.get("ttft_p50"),
                record.get("ttft_p95"),
                record.get("tpot_p50"),
                record.get("tpot_p95"),
                record.get("tpot_unit_sources"),
                record.get("latency_p50"),
                record.get("latency_p95"),
                record.get("all_latency_p95"),
                record.get("queue_p95"),
                record.get("end_to_end_p95"),
                record.get("output_units_per_s") if len(record.get("output_unit_sources") or []) == 1 else None,
                record.get("output_unit_sources"),
                record.get("output_units_per_s_by_source"),
                [record.get("error")] + (record.get("errors") or []) if record.get("error") else record.get("errors"),
                record.get("notes"),
            ]
        )

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
