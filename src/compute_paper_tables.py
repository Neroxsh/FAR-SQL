#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, mean, percentile, read_jsonl, update_manifest, write_json


DETERMINED = {"equivalent", "non_equivalent"}
FALLBACK_FAILURE_STATUSES = {
    "timeout",
    "unsupported_static",
    "unsupported_runtime",
    "unknown",
    "conversion_error",
    "runtime_error",
    "profile_bypass_low_coverage",
    "zero_budget_routed",
}


def read_if_exists(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def pct(x: float) -> str:
    return f"{x:.2%}"


def label_from_status(status: str) -> Optional[str]:
    if status == "equivalent":
        return "yes"
    if status == "non_equivalent":
        return "no"
    return None


def summarize_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    verified = sum(1 for r in rows if r.get("decision_source") == "verified")
    assisted = sum(1 for r in rows if r.get("decision_source") == "model_assisted")
    abstained = sum(1 for r in rows if r.get("decision_source") == "abstained")
    decided = verified + assisted
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    wrong = sum(1 for r in rows if r.get("is_wrong_decision") is True)
    runtimes = [float(r.get("total_runtime", 0.0) or 0.0) for r in rows]
    return {
        "samples": n,
        "verified_coverage": verified / n if n else 0.0,
        "assisted_coverage": assisted / n if n else 0.0,
        "total_coverage": decided / n if n else 0.0,
        "abstention_rate": abstained / n if n else 0.0,
        "accuracy_on_decided": correct / decided if decided else 0.0,
        "wrong_decisions": wrong,
        "wrong_decision_rate": wrong / n if n else 0.0,
        "avg_runtime": mean(runtimes),
        "p50_runtime": percentile(runtimes, 50),
        "p95_runtime": percentile(runtimes, 95),
    }


def stage_as_decisions(stage_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in stage_rows:
        label = label_from_status(str(row.get("verieql_status")))
        final_label = label or "uncertain"
        decided = final_label in {"yes", "no"}
        out.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "gold_label": row.get("gold_label"),
                "final_label": final_label,
                "decision_source": "verified" if label else "abstained",
                "total_runtime": float(row.get("verieql_runtime", row.get("runtime_s", 0.0)) or 0.0),
                "is_decided": decided,
                "is_correct": (final_label == row.get("gold_label")) if decided else None,
                "is_wrong_decision": bool(decided and final_label != row.get("gold_label")),
                "verieql_status": row.get("verieql_status"),
            }
        )
    return out


def summarize_stage(stage_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(stage_rows)
    status_counts = Counter(str(r.get("verieql_status")) for r in stage_rows)
    determined = status_counts.get("equivalent", 0) + status_counts.get("non_equivalent", 0)
    runtimes = [float(r.get("verieql_runtime", r.get("runtime_s", 0.0)) or 0.0) for r in stage_rows]
    return {
        "samples": n,
        "verified_coverage": determined / n if n else 0.0,
        "fallback_ratio": 1.0 - (determined / n) if n else 0.0,
        "equivalent": status_counts.get("equivalent", 0),
        "non_equivalent": status_counts.get("non_equivalent", 0),
        "timeout": status_counts.get("timeout", 0),
        "unsupported": status_counts.get("unsupported_static", 0) + status_counts.get("unsupported_runtime", 0),
        "zero_budget": status_counts.get("profile_bypass_low_coverage", 0) + status_counts.get("zero_budget_routed", 0),
        "unknown": status_counts.get("unknown", 0),
        "conversion_error": status_counts.get("conversion_error", 0),
        "runtime_error": status_counts.get("runtime_error", 0),
        "avg_runtime": mean(runtimes),
        "p95_runtime": percentile(runtimes, 95),
    }


def parse_model_strategy(log: Dict[str, Any], strategy: str) -> Optional[str]:
    parsed = list(log.get("parsed_answers", []))
    if strategy == "always_output":
        return parsed[0] if parsed and parsed[0] in {"yes", "no"} else None
    valid = [x for x in parsed if x in {"yes", "no"}]
    if len(valid) != len(parsed) or not valid:
        return None
    if len(set(valid)) != 1:
        return None
    return valid[0]


def model_only_decisions(prompts: List[Dict[str, Any]], logs: Dict[str, Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    rows = []
    for item in prompts:
        log = logs.get(item["id"])
        label = parse_model_strategy(log, strategy) if log else None
        final_label = label or "uncertain"
        decided = final_label in {"yes", "no"}
        rows.append(
            {
                "id": item["id"],
                "dataset": item["dataset"],
                "gold_label": item["gold_label"],
                "final_label": final_label,
                "decision_source": "model_assisted" if label else "abstained",
                "total_runtime": float(log.get("fallback_runtime", 0.0)) if log else 0.0,
                "is_decided": decided,
                "is_correct": (final_label == item["gold_label"]) if decided else None,
                "is_wrong_decision": bool(decided and final_label != item["gold_label"]),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, rows: List[Dict[str, Any]], columns: List[str], percent_cols: Iterable[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    percent = set(percent_cols)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:" for _ in columns[1:]]) + "|\n")
        for row in rows:
            vals = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    vals.append(pct(value) if col in percent else f"{value:.2f}")
                else:
                    vals.append(str(value))
            f.write("| " + " | ".join(vals) + " |\n")


def collect(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    splits = experiment_test_splits(cfg)
    exp = cfg.get("experiments", {})
    official = str(exp.get("official_strategy_name", "official600"))
    autobudget = str(exp.get("autobudget_strategy_name", "autobudget"))
    fixed_budgets = [int(x) for x in exp.get("fixed_budgets_sec", [10, 30, 60, 120])]
    models = [m for m in cfg["fallback"].get("models", []) if m.get("enabled", True)]
    fallback_strategies = [str(x) for x in cfg["fallback"].get("strategies", ["always_output", "self_consistency", "risk_gated"])]
    main_fallback = str(exp.get("main_fallback_strategy", cfg["fallback"].get("default_strategy", "risk_gated")))

    rq1: List[Dict[str, Any]] = []
    rq2: List[Dict[str, Any]] = []
    rq3: List[Dict[str, Any]] = []
    rq4: List[Dict[str, Any]] = []
    rq5: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []

    for split in splits:
        official_stage = read_if_exists(PROJECT_ROOT / "outputs/decisions" / f"{split}.{official}.verieql_stage.jsonl")
        if official_stage:
            rq1.append({"dataset": split, "strategy": "VeriEQL@600s", **summarize_stage(official_stage)})
            row = {"dataset": split, "family": "symbolic", "model": "-", "method": "VeriEQL@600s", **summarize_decisions(stage_as_decisions(official_stage))}
            rq3.append(row)
            all_metrics.append(row)

        for budget in fixed_budgets:
            strategy = f"fixed{budget}s"
            stage = read_if_exists(PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.verieql_stage.jsonl")
            if not stage:
                continue
            rq2.append({"dataset": split, "strategy": f"Fixed@{budget}s", "budget_type": "fixed", **summarize_stage(stage)})
            row = {"dataset": split, "family": "symbolic", "model": "-", "method": f"VeriEQL Fixed@{budget}s", **summarize_decisions(stage_as_decisions(stage))}
            all_metrics.append(row)

        auto_stage = read_if_exists(PROJECT_ROOT / "outputs/decisions" / f"{split}.{autobudget}.verieql_stage.jsonl")
        if auto_stage:
            rq2.append({"dataset": split, "strategy": "AutoBudget", "budget_type": "profile", **summarize_stage(auto_stage)})
            row = {"dataset": split, "family": "symbolic", "model": "-", "method": "VeriEQL AutoBudget", **summarize_decisions(stage_as_decisions(auto_stage))}
            all_metrics.append(row)

        for model in models:
            model_key = str(model["key"])
            prompt_path = PROJECT_ROOT / "outputs/model_only_prompts" / f"{split}.model_only.prompts.jsonl"
            model_log_path = PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl"
            if prompt_path.exists() and model_log_path.exists():
                prompts = read_jsonl(prompt_path)
                logs = {row["id"]: row for row in read_jsonl(model_log_path)}
                for strategy in ["always_output", "self_consistency"]:
                    row = {
                        "dataset": split,
                        "family": "sft_only",
                        "model": model_key,
                        "method": f"SFT-only-{strategy}",
                        **summarize_decisions(model_only_decisions(prompts, logs, strategy)),
                    }
                    rq3.append(row)
                    all_metrics.append(row)

            for vstrategy in [autobudget, *[f"fixed{b}s" for b in fixed_budgets]]:
                for fstrategy in fallback_strategies:
                    final_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{vstrategy}.{model_key}.{fstrategy}.final.jsonl"
                    if not final_path.exists():
                        continue
                    rows = read_jsonl(final_path)
                    method = "Ours AutoBudget" if vstrategy == autobudget else f"Fixed@{vstrategy.removeprefix('fixed').removesuffix('s')}s + SFT"
                    metric_row = {
                        "dataset": split,
                        "family": "hybrid",
                        "model": model_key,
                        "method": f"{method}-{fstrategy}",
                        "verieql_strategy": vstrategy,
                        "fallback_strategy": fstrategy,
                        **summarize_decisions(rows),
                    }
                    all_metrics.append(metric_row)
                    if fstrategy == main_fallback and (vstrategy == autobudget or vstrategy in {"fixed10s", "fixed30s", "fixed60s"}):
                        rq3.append(metric_row)
                    if vstrategy == autobudget:
                        rq4.append(metric_row)
                        by_failure: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                        for row in rows:
                            status = str(row.get("verieql_status"))
                            if status in FALLBACK_FAILURE_STATUSES:
                                by_failure[status].append(row)
                        for status, part in sorted(by_failure.items()):
                            rq5.append(
                                {
                                    "dataset": split,
                                    "model": model_key,
                                    "fallback_strategy": fstrategy,
                                    "failure_type": status,
                                    **summarize_decisions(part),
                                }
                            )

    return {"rq1": rq1, "rq2": rq2, "rq3": rq3, "rq4": rq4, "rq5": rq5, "all": all_metrics}


def write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# NDBC Paper Results Registry\n\n")
        f.write("This directory is the active result area for the cleaned zero-budget + calibrated short-budget protocol.\n\n")
        f.write("| File | Paper Use |\n")
        f.write("|---|---|\n")
        f.write("| `outputs/tables/rq1_verieql_official.md` | RQ1: VeriEQL@600s official strong baseline status/runtime. |\n")
        f.write("| `outputs/tables/rq2_budget_tradeoff.md` | RQ2: Fixed budgets vs AutoBudget coverage-runtime trade-off. |\n")
        f.write("| `outputs/tables/rq3_main_comparison.md` | RQ3: VeriEQL-only, SFT-only, fixed cascade, and ours. |\n")
        f.write("| `outputs/tables/rq4_fallback_ablation.md` | RQ4: Always output, self-consistency, risk-gated fallback. |\n")
        f.write("| `outputs/tables/rq5_failure_type_fallback.md` | RQ5: fallback behavior by VeriEQL failure type. |\n")
        f.write("| `outputs/metrics/all_experiment_metrics.csv` | Machine-readable aggregate table for plotting and paper drafting. |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = collect(cfg)

    write_json(PROJECT_ROOT / "outputs/metrics/paper_tables.json", tables)
    write_csv(PROJECT_ROOT / "outputs/metrics/all_experiment_metrics.csv", tables["all"])
    write_md_table(
        PROJECT_ROOT / "outputs/tables/rq1_verieql_official.md",
        tables["rq1"],
        ["dataset", "strategy", "samples", "verified_coverage", "equivalent", "non_equivalent", "timeout", "unsupported", "zero_budget", "unknown", "conversion_error", "runtime_error", "avg_runtime", "p95_runtime"],
        {"verified_coverage"},
    )
    write_md_table(
        PROJECT_ROOT / "outputs/tables/rq2_budget_tradeoff.md",
        tables["rq2"],
        ["dataset", "strategy", "budget_type", "samples", "verified_coverage", "fallback_ratio", "timeout", "unsupported", "zero_budget", "avg_runtime", "p95_runtime"],
        {"verified_coverage", "fallback_ratio"},
    )
    write_md_table(
        PROJECT_ROOT / "outputs/tables/rq3_main_comparison.md",
        tables["rq3"],
        ["dataset", "family", "model", "method", "samples", "total_coverage", "verified_coverage", "assisted_coverage", "accuracy_on_decided", "wrong_decision_rate", "avg_runtime", "p95_runtime"],
        {"total_coverage", "verified_coverage", "assisted_coverage", "accuracy_on_decided", "wrong_decision_rate"},
    )
    write_md_table(
        PROJECT_ROOT / "outputs/tables/rq4_fallback_ablation.md",
        tables["rq4"],
        ["dataset", "model", "verieql_strategy", "fallback_strategy", "samples", "total_coverage", "accuracy_on_decided", "wrong_decisions", "wrong_decision_rate", "abstention_rate"],
        {"total_coverage", "accuracy_on_decided", "wrong_decision_rate", "abstention_rate"},
    )
    write_md_table(
        PROJECT_ROOT / "outputs/tables/rq5_failure_type_fallback.md",
        tables["rq5"],
        ["dataset", "model", "fallback_strategy", "failure_type", "samples", "total_coverage", "accuracy_on_decided", "wrong_decisions", "wrong_decision_rate", "abstention_rate"],
        {"total_coverage", "accuracy_on_decided", "wrong_decision_rate", "abstention_rate"},
    )
    write_registry(PROJECT_ROOT / "outputs/results/results_registry.md")
    update_manifest(PROJECT_ROOT / "outputs", "compute_paper_tables", {k: len(v) for k, v in tables.items()})
    print(json.dumps({k: len(v) for k, v in tables.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
