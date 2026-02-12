import argparse
import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

VERSION_NAMES = {"v1", "v2", "v3", "v4", "v5"}

SECRET_PATTERNS = {
    "stripe_secret": re.compile(r"\bsk_(?:test|live)_[A-Za-z0-9]{16,}\b"),
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
}

INFERENCE_KEEP_FIELDS = ["task_id", "metadata", "metrics"]
GRADED_KEEP_FIELDS = ["task_id", "metadata", "metrics", "score", "requirement_status"]
GRADED_ERRORS_KEEP_FIELDS = [
    "task_id",
    "metadata",
    "metrics",
    "score",
    "requirement_status",
    "error_classification",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_pattern_counts() -> Dict[str, int]:
    return {name: 0 for name in SECRET_PATTERNS.keys()}


def add_pattern_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def count_secret_hits_in_text(text: str) -> Dict[str, int]:
    counts = init_pattern_counts()
    for name, pattern in SECRET_PATTERNS.items():
        counts[name] = len(pattern.findall(text))
    return counts


def count_secret_hits_row(row: Dict[str, Any]) -> Dict[str, int]:
    dumped = json.dumps(row, ensure_ascii = False)
    return count_secret_hits_in_text(dumped)


def has_hits(counts: Dict[str, int]) -> bool:
    return any(int(value) > 0 for value in counts.values())


def is_jsonl_candidate(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    if path.name.endswith(".progress.jsonl"):
        return False
    return True


def detect_file_class(path: Path) -> str:
    name = path.name
    if name.endswith("_graded_errors.jsonl"):
        return "graded_errors"
    if name.endswith("_graded.jsonl"):
        return "graded"
    return "inference"


def safe_task_id(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("task_id", row.get("task_id", ""))
        return value if isinstance(value, str) else ""
    value = row.get("task_id", "")
    return value if isinstance(value, str) else ""


def sanitize_requirement_status(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def sanitize_error_types(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        if isinstance(item, str) and item not in result:
            result.append(item)
    return result


def sanitize_error_classification(
    value: Any,
    nested_removed: Dict[str, int],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"task_error_types": []}

    if "per_rubric_classification" in value:
        nested_removed["error_classification.per_rubric_classification"] = (
            int(nested_removed.get("error_classification.per_rubric_classification", 0)) + 1
        )

    return {
        "task_error_types": sanitize_error_types(value.get("task_error_types", [])),
    }


def sanitize_metrics(
    value: Any,
    nested_removed: Dict[str, int],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    metrics = copy.deepcopy(value)

    if "memory_error" in metrics:
        metrics.pop("memory_error", None)
        nested_removed["metrics.memory_error"] = int(nested_removed.get("metrics.memory_error", 0)) + 1

    step_scoring = metrics.get("step_scoring", {})
    if isinstance(step_scoring, dict) and "steps" in step_scoring:
        step_scoring.pop("steps", None)
        nested_removed["metrics.step_scoring.steps"] = int(nested_removed.get("metrics.step_scoring.steps", 0)) + 1

    return metrics


def sanitize_row(
    row: Dict[str, Any],
    file_class: str,
) -> Tuple[Dict[str, Any], List[str], Dict[str, int]]:
    removed_columns: List[str] = []
    nested_removed: Dict[str, int] = {}

    if file_class == "graded_errors":
        keep_fields = GRADED_ERRORS_KEEP_FIELDS
    elif file_class == "graded":
        keep_fields = GRADED_KEEP_FIELDS
    else:
        keep_fields = INFERENCE_KEEP_FIELDS

    sanitized: Dict[str, Any] = {}

    for field in keep_fields:
        if field not in row:
            continue
        if field == "metrics":
            sanitized[field] = sanitize_metrics(row.get(field), nested_removed)
        elif field == "error_classification" and file_class == "graded_errors":
            sanitized[field] = sanitize_error_classification(row.get(field), nested_removed)
        elif field == "requirement_status":
            sanitized[field] = sanitize_requirement_status(row.get(field))
        else:
            sanitized[field] = row.get(field)

    task_id = safe_task_id(row)
    if task_id and not sanitized.get("task_id"):
        sanitized["task_id"] = task_id

    if "metadata" not in sanitized or not isinstance(sanitized.get("metadata"), dict):
        metadata = row.get("metadata", {})
        sanitized["metadata"] = metadata if isinstance(metadata, dict) else {}

    if "metrics" not in sanitized or not isinstance(sanitized.get("metrics"), dict):
        sanitized["metrics"] = {}

    if file_class in {"graded", "graded_errors"} and "score" not in sanitized:
        sanitized["score"] = row.get("score", 0)

    if file_class in {"graded", "graded_errors"} and "requirement_status" not in sanitized:
        sanitized["requirement_status"] = sanitize_requirement_status(row.get("requirement_status", []))

    if file_class == "graded_errors" and "error_classification" not in sanitized:
        sanitized["error_classification"] = {"task_error_types": []}

    for key in row.keys():
        if key not in keep_fields:
            removed_columns.append(key)

    return sanitized, removed_columns, nested_removed


def resolve_target_dirs(input_root: Path, version: str) -> List[Path]:
    if version == "all":
        if input_root.name in VERSION_NAMES:
            return [input_root]
        version_dirs = [p for p in input_root.iterdir() if p.is_dir() and p.name in VERSION_NAMES]
        if version_dirs:
            return sorted(version_dirs, key = lambda item: item.name)
        return [input_root]

    if input_root.name == version:
        return [input_root]

    candidate = input_root / version
    if candidate.exists() and candidate.is_dir():
        return [candidate]

    raise ValueError(f"Version directory not found: {version} under {input_root}")


def read_jsonl_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []

    with path.open("r", encoding = "utf-8") as handle:
        for index, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": index, "error": str(exc)})
                continue
            if not isinstance(item, dict):
                parse_errors.append({"line": index, "error": "row is not a JSON object"})
                continue
            rows.append(item)

    return rows, parse_errors


def write_jsonl_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii = False) + "\n")


def process_file(
    path: Path,
    input_root: Path,
    in_place: bool,
) -> Dict[str, Any]:
    file_class = detect_file_class(path)
    rows, parse_errors = read_jsonl_rows(path)

    before_counts_total = init_pattern_counts()
    after_counts_total = init_pattern_counts()
    columns_removed: Dict[str, int] = {}
    nested_removed: Dict[str, int] = {}
    sanitized_rows: List[Dict[str, Any]] = []

    for row in rows:
        before_counts = count_secret_hits_row(row)
        add_pattern_counts(before_counts_total, before_counts)

        sanitized_row, removed_columns, nested_removed_row = sanitize_row(row, file_class)
        sanitized_rows.append(sanitized_row)

        for column in removed_columns:
            columns_removed[column] = int(columns_removed.get(column, 0)) + 1

        for key, value in nested_removed_row.items():
            nested_removed[key] = int(nested_removed.get(key, 0)) + int(value)

        after_counts = count_secret_hits_row(sanitized_row)
        add_pattern_counts(after_counts_total, after_counts)

    updated = False
    if in_place and not parse_errors:
        write_jsonl_rows(path, sanitized_rows)
        updated = True

    return {
        "path": str(path),
        "file_class": file_class,
        "rows": len(rows),
        "parse_errors": parse_errors,
        "columns_removed": columns_removed,
        "nested_fields_removed": nested_removed,
        "secret_hits_before": before_counts_total,
        "secret_hits_after": after_counts_total,
        "updated": updated,
    }


def scan_files(input_root: Path, version: str) -> List[Path]:
    targets = resolve_target_dirs(input_root, version)
    files: List[Path] = []
    for target in targets:
        for path in sorted(target.rglob("*.jsonl")):
            if is_jsonl_candidate(path):
                files.append(path)
    return files


def build_report(
    input_root: Path,
    version: str,
    in_place: bool,
    mode: str,
    file_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_rows = 0
    columns_removed_total: Dict[str, int] = {}
    nested_removed_total: Dict[str, int] = {}
    before_total = init_pattern_counts()
    after_total = init_pattern_counts()
    parse_errors: List[Dict[str, Any]] = []

    for file_report in file_reports:
        total_rows += int(file_report.get("rows", 0))
        add_pattern_counts(before_total, file_report.get("secret_hits_before", {}))
        add_pattern_counts(after_total, file_report.get("secret_hits_after", {}))

        for key, value in file_report.get("columns_removed", {}).items():
            columns_removed_total[key] = int(columns_removed_total.get(key, 0)) + int(value)

        for key, value in file_report.get("nested_fields_removed", {}).items():
            nested_removed_total[key] = int(nested_removed_total.get(key, 0)) + int(value)

        for parse_error in file_report.get("parse_errors", []):
            parse_errors.append({"path": file_report.get("path", ""), **parse_error})

    has_after_hits = has_hits(after_total)
    has_parse_errors = len(parse_errors) > 0

    if mode == "strict" and (has_after_hits or has_parse_errors):
        status = "failed"
    elif has_after_hits or has_parse_errors:
        status = "warn"
    else:
        status = "ok"

    return {
        "generated_at": utc_now_iso(),
        "input_root": str(input_root),
        "version": version,
        "in_place": bool(in_place),
        "mode": mode,
        "status": status,
        "files_processed": len(file_reports),
        "rows_processed": total_rows,
        "columns_removed": columns_removed_total,
        "nested_fields_removed": nested_removed_total,
        "secret_hits_before": before_total,
        "secret_hits_after": after_total,
        "parse_errors": parse_errors,
        "files": file_reports,
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as handle:
        json.dump(payload, handle, indent = 2, ensure_ascii = False)


def main() -> None:
    parser = argparse.ArgumentParser(description = "Sanitize benchmark result JSONL artifacts")
    parser.add_argument("--input-root", type = str, default = "benchmark/results")
    parser.add_argument("--version", type = str, default = "all", choices = ["v1", "v2", "v3", "v4", "v5", "all"])
    parser.add_argument("--in-place", action = argparse.BooleanOptionalAction, default = False)
    parser.add_argument("--report-path", type = str, default = "benchmark/results/sanitize_report.json")
    parser.add_argument("--mode", type = str, default = "warn", choices = ["warn", "strict"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root not found or not a directory: {input_root}")

    files = scan_files(input_root, args.version)
    file_reports: List[Dict[str, Any]] = []

    for path in files:
        report = process_file(path, input_root, bool(args.in_place))
        file_reports.append(report)

    report_payload = build_report(
        input_root = input_root,
        version = args.version,
        in_place = bool(args.in_place),
        mode = args.mode,
        file_reports = file_reports,
    )
    write_report(Path(args.report_path), report_payload)

    files_processed = report_payload["files_processed"]
    rows_processed = report_payload["rows_processed"]
    status = report_payload["status"]
    print(f"Sanitization report: {args.report_path}")
    print(f"Files processed: {files_processed}, rows processed: {rows_processed}")
    print(f"Status: {status}")

    if args.mode == "strict" and report_payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
