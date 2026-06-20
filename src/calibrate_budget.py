#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    PROJECT_ROOT,
    load_config,
    percentile,
    read_jsonl,
    sql_pair_hash,
    standard_paths,
    update_manifest,
    verieql_candidate_budgets,
    verieql_reference_budget,
    write_json,
    write_jsonl,
)
from extract_features import assign_bucket, features_one, flags_for_pair, merge_features
from normalize_verieql import normalize_status, raw_runtime
from routing_policy import choose_budget, determined_coverage


BUCKETS = ["Simple-SPJ", "Aggregation", "Nested/Set", "Complex-Mixed"]
DETERMINED = {"equivalent", "non_equivalent"}


def coverage(rows: List[Dict[str, Any]], budget: float) -> float:
    return determined_coverage(rows, budget)


def load_test_hashes(cfg: Dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for path in standard_paths(cfg).values():
        if not path.exists():
            continue
        for row in read_jsonl(path):
            hashes.add(sql_pair_hash(row["sql1"], row["sql2"], row.get("schema")))
    return hashes


def normalize_record(row: Dict[str, Any], row_no: int, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pair = row.get("pair")
    if not isinstance(pair, list) or len(pair) < 2 or not isinstance(pair[0], str) or not isinstance(pair[1], str):
        return None
    std = {
        "id": f"calibration:{row_no}",
        "dataset": "calibration",
        "schema": row.get("schema", {}),
        "constraint": row.get("constraint"),
        "sql1": pair[0],
        "sql2": pair[1],
    }
    f1 = features_one(std["sql1"])
    f2 = features_one(std["sql2"])
    features = merge_features(f1, f2, std)
    bucket = assign_bucket(features)
    static_flags, risk_flags = flags_for_pair(std["sql1"], std["sql2"])
    status, label = normalize_status(row)
    runtime = raw_runtime(row, status, float(verieql_reference_budget(cfg)))
    return {
        "id": std["id"],
        "dataset": "calibration",
        "source_name": Path(str(cfg["data_sources"]["calibration"]["path"])).name,
        "source_line": row_no,
        "source_index": row.get("index"),
        "source_file": row.get("file"),
        "features": features,
        "bucket": bucket,
        "static_unsupported_flags": static_flags,
        "risk_flags": risk_flags,
        "raw_states": row.get("states", []),
        "raw_err": row.get("err"),
        "normalized_status": status,
        "verieql_label": label,
        "runtime_s": runtime,
    }


def write_profile_tables(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    budgets = verieql_candidate_budgets(cfg)
    reference_budget = verieql_reference_budget(cfg)
    epsilon = float(cfg["budget"]["epsilon"])
    min_cov = float(cfg["budget"]["min_coverage_threshold"])
    default_budget = int(cfg["budget"]["default_budget_sec"])

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    budget_table = {}
    profile_rows = []
    for bucket in BUCKETS:
        part = by_bucket.get(bucket, [])
        status_counts = Counter(r["normalized_status"] for r in part)
        selected = choose_budget(part, budgets, epsilon, min_cov, default_budget)
        budget_table[bucket] = selected
        profile_rows.append(
            {
                "Bucket": bucket,
                "#Samples": len(part),
                **{f"Det.@{b}s": coverage(part, b) for b in budgets},
                f"Timeout@{reference_budget}s": status_counts.get("timeout", 0) / len(part) if part else 0.0,
                "Unsupported": (status_counts.get("unsupported_static", 0) + status_counts.get("unsupported_runtime", 0)) / len(part) if part else 0.0,
                "ConversionError": status_counts.get("conversion_error", 0) / len(part) if part else 0.0,
                "RuntimeError": status_counts.get("runtime_error", 0) / len(part) if part else 0.0,
                "Unknown": status_counts.get("unknown", 0) / len(part) if part else 0.0,
                f"P95@{reference_budget}s": percentile([float(r["runtime_s"]) for r in part], 95),
                "SelectedBudget": selected,
            }
        )

    out_dir = PROJECT_ROOT / "outputs/budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "budget_table.json", budget_table)
    write_json(out_dir / "calibration_profile.json", profile_rows)
    csv_path = out_dir / "calibration_profile.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    md_path = PROJECT_ROOT / "outputs/tables/rq1_calibration_profile.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        det_cols = [f"Det.@{b}s" for b in budgets]
        header = ["Bucket", "#Samples", *det_cols, f"Timeout@{reference_budget}s", "Unsupported", "Conversion Error", "Runtime Error", "Unknown", f"P95@{reference_budget}s", "Selected Budget"]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|---|---:" + "|---:" * (len(header) - 2) + "|\n")
        for r in profile_rows:
            values = [r["Bucket"], str(r["#Samples"])]
            values.extend(f"{r[col]:.2%}" for col in det_cols)
            values.extend(
                [
                    f"{r[f'Timeout@{reference_budget}s']:.2%}",
                    f"{r['Unsupported']:.2%}",
                    f"{r['ConversionError']:.2%}",
                    f"{r['RuntimeError']:.2%}",
                    f"{r['Unknown']:.2%}",
                    f"{r[f'P95@{reference_budget}s']:.2f}",
                    str(r["SelectedBudget"]),
                ]
            )
            f.write("| " + " | ".join(values) + " |\n")
    return {"budget_table": budget_table, "profile_rows": profile_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    source = Path(cfg["data_sources"]["calibration"]["path"])
    excluded = load_test_hashes(cfg)
    rows = []
    skipped_overlap = 0
    for row_no, row in enumerate(read_jsonl(source), 1):
        pair = row.get("pair")
        if isinstance(pair, list) and len(pair) >= 2:
            h = sql_pair_hash(pair[0], pair[1], row.get("schema", {}))
            if h in excluded:
                skipped_overlap += 1
                continue
        normalized = normalize_record(row, row_no, cfg)
        if normalized is not None:
            rows.append(normalized)
        if args.limit and len(rows) >= args.limit:
            break

    profile_path = PROJECT_ROOT / "outputs/calibration/calibration_verieql_profile.jsonl"
    profile_rows = write_jsonl(profile_path, rows)
    tables = write_profile_tables(rows, cfg)
    payload = {
        "source": str(source),
        "profile_path": str(profile_path),
        "profile_rows": profile_rows,
        "skipped_test_overlap": skipped_overlap,
        **tables,
    }
    update_manifest(PROJECT_ROOT / "outputs", "calibrate_budget", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
