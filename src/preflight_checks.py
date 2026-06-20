#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import PROJECT_ROOT, load_config, parse_answer, read_jsonl, write_json
from convert_verieql_ready import convert_schema
from prepare_data import standardize_test_row


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def head_jsonl(path: Path, n: int = 2) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_report(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    info: Dict[str, Any] = {"path": path_text, "exists": path.exists(), "is_dir": path.is_dir()}
    if path.exists() and path.is_file() and path.suffix in {".jsonl", ".out"}:
        rows = head_jsonl(path)
        info.update(
            {
                "rows": count_lines(path),
                "head_keys": [list(row.keys()) for row in rows],
                "head_schema_type": [type(row.get("schema")).__name__ for row in rows],
            }
        )
        if rows:
            info["first_states"] = rows[0].get("states")
            info["first_label"] = rows[0].get("semantic equivalence", rows[0].get("label"))
    return info


def check_answer_parser() -> Dict[str, Any]:
    cases = {
        "last_valid_tag_after_placeholder": ("Example <answer>yes/no</answer>\nReasoning...\n<answer>no</answer>", "no"),
        "last_valid_tag_wins": ("<answer>yes</answer>\nLater correction\n<answer>no</answer>", "no"),
        "final_answer_fallback": ("After analysis. Final answer: yes", "yes"),
        "invalid_tag_parse_failed": ("<answer>yes/no</answer>", None),
    }
    results = {}
    ok = True
    for name, (text, expected) in cases.items():
        got = parse_answer(text)
        results[name] = {"expected": expected, "got": got, "ok": got == expected}
        ok = ok and got == expected
    return {"ok": ok, "cases": results}


def check_conversion(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split_name, split_spec in cfg["data_sources"].items():
        if "sources" not in split_spec:
            continue
        split_rows = []
        for source in split_spec["sources"]:
            path = source["path"]
            rows = read_jsonl(path)
            if not rows:
                split_rows.append({"path": path, "ok": False, "error": "empty_source"})
                continue
            try:
                std = standardize_test_row(rows[0], split_spec["dataset"], path, source.get("label"), 1)
                if std is None:
                    raise ValueError("first row could not be standardized")
                schema, constraints = convert_schema(std)
                split_rows.append(
                    {
                        "path": path,
                        "ok": True,
                        "standard_id": std["id"],
                        "tables": len(schema),
                        "first_tables": list(schema)[:5],
                        "constraints": len(constraints),
                    }
                )
            except Exception as exc:
                split_rows.append({"path": path, "ok": False, "error": str(exc)})
        out[split_name] = split_rows
    return out


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# NDBC Preflight Report\n\n")
        f.write("This report is generated before long runs to prevent stale assumptions about files, schemas, parser behavior, and model paths.\n\n")
        f.write("## Sources\n\n")
        f.write("| Name | Exists | Rows | Head Keys | Schema Type |\n")
        f.write("|---|---:|---:|---|---|\n")
        for name, info in report["sources"].items():
            f.write(
                f"| {name} | {info.get('exists')} | {info.get('rows', '')} | "
                f"{info.get('head_keys', '')} | {info.get('head_schema_type', '')} |\n"
            )
        f.write("\n## Models\n\n")
        f.write("| Key | Exists | Path |\n")
        f.write("|---|---:|---|\n")
        for model in report["models"]:
            f.write(f"| {model['key']} | {model['exists']} | `{model['path']}` |\n")
        f.write("\n## Answer Parser\n\n")
        f.write(f"Overall: `{report['answer_parser']['ok']}`\n\n")
        f.write("## Schema Conversion Smoke\n\n")
        f.write("| Split | Source | OK | Tables | First Tables | Error |\n")
        f.write("|---|---|---:|---:|---|---|\n")
        for split, rows in report["schema_conversion"].items():
            for row in rows:
                f.write(
                    f"| {split} | `{row['path']}` | {row.get('ok')} | {row.get('tables', '')} | "
                    f"{row.get('first_tables', '')} | {row.get('error', '')} |\n"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    sources: Dict[str, Any] = {}
    calibration = cfg["data_sources"]["calibration"]
    sources["calibration"] = source_report(calibration["path"])
    for split_name, spec in cfg["data_sources"].items():
        for idx, source in enumerate(spec.get("sources", []), 1):
            sources[f"{split_name}:{idx}"] = source_report(source["path"])

    models = []
    for model in cfg["fallback"].get("models", []):
        path = Path(model["path"])
        models.append({"key": model["key"], "path": str(path), "exists": path.exists(), "is_dir": path.is_dir()})

    report = {
        "sources": sources,
        "models": models,
        "answer_parser": check_answer_parser(),
        "schema_conversion": check_conversion(cfg),
        "config_summary": {
            "candidate_budgets_sec": cfg["verieql"].get("candidate_budgets_sec"),
            "official_timeout_sec": cfg["verieql"].get("official_timeout_sec"),
            "sample_bound": cfg["verieql"].get("sample_bound"),
            "fallback_models": [m["key"] for m in cfg["fallback"].get("models", []) if m.get("enabled", True)],
        },
    }
    write_json(PROJECT_ROOT / "outputs/preflight/preflight_report.json", report)
    write_markdown(PROJECT_ROOT / "outputs/preflight/preflight_report.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
