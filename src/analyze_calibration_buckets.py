#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

from common import PROJECT_ROOT, load_config, verieql_candidate_budgets, verieql_reference_budget, write_json
from extract_features import assign_bucket, features_one, flags_for_pair, merge_features
from normalize_verieql import normalize_status, raw_runtime


DETERMINED = {"equivalent", "non_equivalent"}


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def problem_id(file_text: Any) -> str:
    m = re.search(r"raw_data/(.+?)\.csv", str(file_text))
    return m.group(1) if m else str(file_text)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = json.loads(line)
            pair = raw.get("pair") or ["", ""]
            sql1 = pair[0] if len(pair) > 0 else ""
            sql2 = pair[1] if len(pair) > 1 else ""
            std = {"sql1": sql1, "sql2": sql2, "schema": raw.get("schema"), "constraint": raw.get("constraint")}
            features = merge_features(features_one(sql1), features_one(sql2), std)
            static_flags, risk_flags = flags_for_pair(sql1, sql2)
            status, label = normalize_status(raw)
            rows.append(
                {
                    "line_no": line_no,
                    "raw_index": raw.get("index"),
                    "source_file": raw.get("file"),
                    "problem": problem_id(raw.get("file")),
                    "states": raw.get("states", []),
                    "status": status,
                    "label": label,
                    "runtime_s": raw_runtime(raw, status, 120.0),
                    "bucket": assign_bucket(features),
                    "features": features,
                    "static_unsupported_flags": static_flags,
                    "risk_flags": risk_flags,
                    "err": raw.get("err"),
                }
            )
    return rows


def coverage(part: List[Dict[str, Any]], budget: int) -> float:
    if not part:
        return 0.0
    return sum(1 for row in part if row["status"] in DETERMINED and float(row["runtime_s"]) <= budget) / len(part)


def summarize_group(name: str, part: List[Dict[str, Any]], budgets: List[int]) -> Dict[str, Any]:
    det = [float(row["runtime_s"]) for row in part if row["status"] in DETERMINED]
    timeouts = [float(row["runtime_s"]) for row in part if row["status"] == "timeout"]
    min_budget = min(budgets) if budgets else 1
    reference_budget = max(budgets) if budgets else 120
    return {
        "name": name,
        "samples": len(part),
        "status_counts": dict(Counter(row["status"] for row in part)),
        **{f"det_at_{b}s": coverage(part, b) for b in budgets},
        "gain_120_minus_10": coverage(part, 120) - coverage(part, 10),
        "gain_reference_minus_min": coverage(part, reference_budget) - coverage(part, min_budget),
        "determined_count": len(det),
        "determined_runtime": {
            "min": min(det) if det else 0.0,
            "p50": percentile(det, 50),
            "p90": percentile(det, 90),
            "p95": percentile(det, 95),
            "p99": percentile(det, 99),
            "max": max(det) if det else 0.0,
        },
        "timeout_count": len(timeouts),
        "timeout_runtime": {
            "min": min(timeouts) if timeouts else 0.0,
            "p50": percentile(timeouts, 50),
            "p95": percentile(timeouts, 95),
            "max": max(timeouts) if timeouts else 0.0,
        },
    }


def fine_feature_key(row: Dict[str, Any]) -> str:
    f = row["features"]
    return "|".join(
        [
            "agg" if f["has_aggregation"] or f["has_group_by"] or f["has_having"] else "noagg",
            "nested" if f["nesting_depth"] > 0 else "nonested",
            "set" if f["has_set_op"] else "noset",
            "join3+" if f["join_count"] >= 3 else ("join1-2" if f["join_count"] > 0 else "join0"),
            "tables4+" if f["table_count"] >= 4 else ("tables2-3" if f["table_count"] >= 2 else "table1"),
            "static" if row["static_unsupported_flags"] else "nostatic",
            "risk" if row["risk_flags"] else "norisk",
        ]
    )


def length_complexity_key(row: Dict[str, Any]) -> str:
    f = row["features"]
    return "|".join(
        [
            row["bucket"],
            f"len:{min(int(f['sql_length']) // 50, 6) * 50}+",
            f"pred:{min(int(f['predicate_count']) // 5, 5) * 5}+",
            f"nest:{min(int(f['nesting_depth']), 3)}",
            f"join:{min(int(f['join_count']), 4)}",
            "static" if row["static_unsupported_flags"] else "nostatic",
        ]
    )


def top_group_summaries(
    rows: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
    min_n: int,
    top_n: int,
    budgets: List[int],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    summaries = [summarize_group(name, part, budgets) for name, part in groups.items() if len(part) >= min_n]
    by_gain = sorted(summaries, key=lambda x: (x["gain_120_minus_10"], x["det_at_120s"], x["samples"]), reverse=True)[:top_n]
    by_cov = sorted(summaries, key=lambda x: (x["det_at_120s"], x["samples"]), reverse=True)[:top_n]
    return {"by_gain_120_minus_10": by_gain, "by_det_at_120s": by_cov, "num_groups_kept": len(summaries)}


def choose_budget(part: List[Dict[str, Any]], budgets: List[int], epsilon: float, min_cov: float) -> int:
    if not part:
        return min(budgets)
    max_cov = coverage(part, max(budgets))
    if max_cov < min_cov:
        return 0
    for budget in budgets:
        if max_cov - coverage(part, budget) <= epsilon:
            return budget
    return max(budgets)


def time_distribution(rows: List[Dict[str, Any]], budgets: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    groups = {
        "all_determined": [row for row in rows if row["status"] in DETERMINED],
        "equivalent": [row for row in rows if row["status"] == "equivalent"],
        "non_equivalent": [row for row in rows if row["status"] == "non_equivalent"],
    }
    for name, part in groups.items():
        runtimes = sorted(float(row["runtime_s"]) for row in part)
        if not runtimes:
            out[name] = {"count": 0}
            continue
        pct = {}
        for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 100]:
            pct[str(q)] = percentile(runtimes, q)
        out[name] = {
            "count": len(runtimes),
            "percentiles_s": pct,
            "coverage_by_budget_overall": {str(b): sum(x <= b for x in runtimes) / len(rows) for b in budgets},
            "coverage_by_budget_among_determined": {str(b): sum(x <= b for x in runtimes) / len(runtimes) for b in budgets},
        }
    return out


def make_report(
    rows: List[Dict[str, Any]],
    min_group_size: int,
    top_n: int,
    budgets: List[int],
    epsilon: float,
    min_cov: float,
) -> Dict[str, Any]:
    state_patterns = Counter(tuple(row["states"]) for row in rows)
    bucket_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    problem_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_groups[row["bucket"]].append(row)
        problem_groups[row["problem"]].append(row)

    long_gain_examples = [
        {
            "line_no": row["line_no"],
            "raw_index": row["raw_index"],
            "bucket": row["bucket"],
            "problem": row["problem"],
            "status": row["status"],
            "runtime_s": row["runtime_s"],
            "states": row["states"],
            "features": row["features"],
            "static_unsupported_flags": row["static_unsupported_flags"],
            "risk_flags": row["risk_flags"],
            "err": row["err"],
        }
        for row in rows
        if row["status"] in DETERMINED and float(row["runtime_s"]) > 10
    ][:30]

    partial_equ_timeout_examples = [
        {
            "line_no": row["line_no"],
            "raw_index": row["raw_index"],
            "bucket": row["bucket"],
            "problem": row["problem"],
            "runtime_s": row["runtime_s"],
            "states": row["states"][:20],
            "features": row["features"],
            "static_unsupported_flags": row["static_unsupported_flags"],
            "risk_flags": row["risk_flags"],
        }
        for row in rows
        if "TMO" in row["states"] and "EQU" in row["states"] and "NEQ" not in row["states"]
    ][:30]

    return {
        "samples": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "label_counts": {str(k): v for k, v in Counter((row["status"], row["label"]) for row in rows).items()},
        "state_patterns_top": [{"states": list(k), "count": v} for k, v in state_patterns.most_common(40)],
        "all_equ_pattern_count": sum(v for k, v in state_patterns.items() if k and all(x == "EQU" for x in k)),
        "contains_neq_count": sum(v for k, v in state_patterns.items() if "NEQ" in k),
        "contains_tmo_without_neq_count": sum(v for k, v in state_patterns.items() if "TMO" in k and "NEQ" not in k),
        "budget_candidates_sec": budgets,
        "budget_selection_rule": {
            "epsilon": epsilon,
            "min_coverage_threshold": min_cov,
            "zero_budget_action": 0,
            "quick_probe_budget": min(budgets),
            "reference_budget": max(budgets),
        },
        "time_distribution": time_distribution(rows, budgets),
        "bucket_summary": [
            {**summarize_group(name, part, budgets), "selected_budget": choose_budget(part, budgets, epsilon, min_cov)}
            for name, part in sorted(bucket_groups.items())
        ],
        "fine_feature_groups": top_group_summaries(rows, fine_feature_key, min_group_size, top_n, budgets),
        "length_complexity_groups": top_group_summaries(rows, length_complexity_key, min_group_size, top_n, budgets),
        "problem_groups": top_group_summaries(rows, lambda row: row["problem"], max(20, min_group_size // 2), top_n, budgets),
        "long_gain_examples": long_gain_examples,
        "partial_equ_timeout_examples": partial_equ_timeout_examples,
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    budgets = [int(x) for x in report["budget_candidates_sec"]]
    compact_budgets = [b for b in budgets if b in {1, 2, 3, 5, 10, 120}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Calibration Bucket Audit\n\n")
        f.write("This audit checks whether the bucket design is too coarse and whether fine-grained integer-second budgets create meaningful latency/coverage differences.\n\n")
        f.write(f"Samples: `{report['samples']}`\n\n")
        f.write(f"Status counts: `{report['status_counts']}`\n\n")
        f.write(f"All-EQU successful patterns: `{report['all_equ_pattern_count']}`\n\n")
        f.write(f"Budget candidates: `{budgets}`\n\n")
        f.write(f"Budget selection rule: `{report['budget_selection_rule']}`\n\n")
        f.write("## Runtime Distribution\n\n")
        f.write("| Group | #Determined | P50 | P90 | P95 | P97 | P99 | P99.5 | P99.9 | Max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for group_name, row in report["time_distribution"].items():
            p = row.get("percentiles_s", {})
            f.write(
                f"| {group_name} | {row.get('count', 0)} | {p.get('50', 0.0):.3f} | {p.get('90', 0.0):.3f} | "
                f"{p.get('95', 0.0):.3f} | {p.get('97', 0.0):.3f} | {p.get('99', 0.0):.3f} | "
                f"{p.get('99.5', 0.0):.3f} | {p.get('99.9', 0.0):.3f} | {p.get('100', 0.0):.3f} |\n"
            )
        f.write("## Bucket Summary\n\n")
        det_cols = [f"Det@{b}" for b in compact_budgets]
        f.write("| Bucket | #Samples | " + " | ".join(det_cols) + " | Selected | GainRef-Min | Status Counts |\n")
        f.write("|---|---:" + "|---:" * len(det_cols) + "|---:|---:|---|\n")
        for row in report["bucket_summary"]:
            vals = [f"{row[f'det_at_{b}s']:.2%}" for b in compact_budgets]
            f.write(f"| {row['name']} | {row['samples']} | " + " | ".join(vals) + f" | {row['selected_budget']} | {row['gain_reference_minus_min']:.2%} | `{row['status_counts']}` |\n")
        for section, title in [
            ("fine_feature_groups", "Fine Feature Groups"),
            ("length_complexity_groups", "Length/Complexity Groups"),
            ("problem_groups", "Problem Groups"),
        ]:
            f.write(f"\n## {title}: Largest 120s-10s Gain\n\n")
            f.write("| Group | #Samples | Det@1 | Det@2 | Det@10 | Det@120 | Gain120-10 | Status Counts |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
            for row in report[section]["by_gain_120_minus_10"][:20]:
                f.write(
                    f"| {row['name']} | {row['samples']} | {row.get('det_at_1s', 0.0):.2%} | {row.get('det_at_2s', 0.0):.2%} | "
                    f"{row.get('det_at_10s', 0.0):.2%} | {row.get('det_at_120s', 0.0):.2%} | {row['gain_120_minus_10']:.2%} | "
                    f"`{row['status_counts']}` |\n"
                )
        f.write("\n## Top Raw State Patterns\n\n")
        f.write("| Count | States |\n")
        f.write("|---:|---|\n")
        for row in report["state_patterns_top"][:25]:
            f.write(f"| {row['count']} | `{row['states']}` |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--min-group-size", type=int, default=80)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    source = Path(cfg["data_sources"]["calibration"]["path"])
    budgets = verieql_candidate_budgets(cfg)
    reference_budget = verieql_reference_budget(cfg)
    audit_budgets = sorted(set(budgets + [reference_budget]))
    rows = load_rows(source)
    report = make_report(
        rows,
        args.min_group_size,
        args.top_n,
        audit_budgets,
        float(cfg["budget"]["epsilon"]),
        float(cfg["budget"]["min_coverage_threshold"]),
    )
    write_json(PROJECT_ROOT / "outputs/budget/calibration_bucket_audit.json", report)
    write_markdown(PROJECT_ROOT / "outputs/tables/calibration_bucket_audit.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
