#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, mean, percentile, read_jsonl, write_json
from fuse_decisions import fallback_label, label_from_verieql_status


FALLBACK_STRATEGIES = ["always_output", "self_consistency", "majority", "risk_gated"]


def load_jsonl_or_empty(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def load_standard(split: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(PROJECT_ROOT / "data/standard" / f"{split}.jsonl")}


def load_stage(split: str, strategy: str) -> List[Dict[str, Any]]:
    return load_jsonl_or_empty(PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.verieql_stage.jsonl")


def load_model_logs(split: str, model_key: str, stage_strategy: Optional[str] = None, suffix: str = "model_only") -> Dict[str, Dict[str, Any]]:
    if stage_strategy is None or suffix == "model_only":
        path = PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl"
    else:
        path = PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{stage_strategy}.{model_key}.{suffix}.jsonl"
    return {str(row["id"]): row for row in load_jsonl_or_empty(path)}


def evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    decided = [r for r in rows if r["is_decided"]]
    correct = [r for r in rows if r["is_correct"] is True]
    wrong = [r for r in rows if r["is_decided"] and r["is_correct"] is False]
    runtimes = [float(r.get("formal_runtime", 0.0) or 0.0) for r in rows]
    total_runtimes = [float(r.get("total_runtime", r.get("formal_runtime", 0.0)) or 0.0) for r in rows]
    return {
        "samples": n,
        "overall_acc": len(correct) / n if n else 0.0,
        "coverage": len(decided) / n if n else 0.0,
        "accuracy_on_decided": len(correct) / len(decided) if decided else 0.0,
        "wrong_rate": len(wrong) / n if n else 0.0,
        "abstention_rate": 1.0 - (len(decided) / n if n else 0.0),
        "correct": len(correct),
        "wrong": len(wrong),
        "decided": len(decided),
        "avg_formal_runtime": mean(runtimes),
        "p95_formal_runtime": percentile(runtimes, 95),
        "avg_total_runtime": mean(total_runtimes),
        "p95_total_runtime": percentile(total_runtimes, 95),
        "decision_sources": dict(Counter(str(r.get("decision_source")) for r in rows)),
    }


def evaluate_formal_only(split: str, strategy: str, stage_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for stage in stage_rows:
        label = label_from_verieql_status(str(stage.get("verieql_status")))
        rows.append(
            {
                "id": stage["id"],
                "gold_label": stage["gold_label"],
                "final_label": label if label in {"yes", "no"} else "uncertain",
                "is_decided": label in {"yes", "no"},
                "is_correct": (label == stage["gold_label"]) if label in {"yes", "no"} else None,
                "formal_runtime": float(stage.get("verieql_runtime", 0.0) or 0.0),
                "total_runtime": float(stage.get("verieql_runtime", 0.0) or 0.0),
                "decision_source": "verified" if label in {"yes", "no"} else "abstained",
            }
        )
    return {
        "dataset": split,
        "method": strategy,
        "category": "formal_only",
        "model": "-",
        "fallback": "-",
        "prompt_variant": "-",
        **evaluate_rows(rows),
    }


def evaluate_hybrid(
    split: str,
    strategy: str,
    stage_rows: List[Dict[str, Any]],
    logs: Dict[str, Dict[str, Any]],
    model_key: str,
    fallback_strategy: str,
    prompt_variant: str,
) -> Dict[str, Any]:
    rows = []
    for stage in stage_rows:
        label = label_from_verieql_status(str(stage.get("verieql_status")))
        source = "verified"
        fallback_runtime = 0.0
        if label is None:
            log = logs.get(str(stage["id"]))
            label = fallback_label(log, fallback_strategy, stage)
            fallback_runtime = float(log.get("fallback_runtime", 0.0) or 0.0) if log else 0.0
            source = "model_assisted" if label in {"yes", "no"} else "abstained"
        formal_runtime = float(stage.get("verieql_runtime", 0.0) or 0.0)
        rows.append(
            {
                "id": stage["id"],
                "gold_label": stage["gold_label"],
                "final_label": label if label in {"yes", "no"} else "uncertain",
                "is_decided": label in {"yes", "no"},
                "is_correct": (label == stage["gold_label"]) if label in {"yes", "no"} else None,
                "formal_runtime": formal_runtime,
                "total_runtime": formal_runtime + fallback_runtime,
                "decision_source": source,
            }
        )
    return {
        "dataset": split,
        "method": strategy,
        "category": "hybrid",
        "model": model_key,
        "fallback": fallback_strategy,
        "prompt_variant": prompt_variant,
        **evaluate_rows(rows),
    }


def evaluate_model_only(split: str, logs: Dict[str, Dict[str, Any]], model_key: str, fallback_strategy: str) -> Dict[str, Any]:
    standard = load_standard(split)
    rows = []
    for sid, sample in standard.items():
        log = logs.get(sid)
        label = fallback_label(log, fallback_strategy, {"verieql_status": "model_only"})
        model_runtime = float(log.get("fallback_runtime", 0.0) or 0.0) if log else 0.0
        rows.append(
            {
                "id": sid,
                "gold_label": sample["label"],
                "final_label": label if label in {"yes", "no"} else "uncertain",
                "is_decided": label in {"yes", "no"},
                "is_correct": (label == sample["label"]) if label in {"yes", "no"} else None,
                "formal_runtime": 0.0,
                "total_runtime": model_runtime,
                "decision_source": "model_only" if label in {"yes", "no"} else "abstained",
            }
        )
    return {
        "dataset": split,
        "method": "model_only",
        "category": "model_only",
        "model": model_key,
        "fallback": fallback_strategy,
        "prompt_variant": "training_compatible",
        **evaluate_rows(rows),
    }


def evaluate_precomputed_final(split: str, strategy: str, path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    meta = rows[0] if rows else {}
    eval_rows = [
        {
            "id": row["id"],
            "gold_label": row["gold_label"],
            "final_label": row["final_label"],
            "is_decided": bool(row["is_decided"]),
            "is_correct": row["is_correct"],
            "formal_runtime": float(row.get("verieql_runtime", 0.0) or 0.0),
            "total_runtime": float(row.get("total_runtime", row.get("verieql_runtime", 0.0)) or 0.0),
            "decision_source": row.get("decision_source"),
        }
        for row in rows
    ]
    return {
        "dataset": split,
        "method": strategy,
        "category": str(meta.get("category", "hybrid_specialist")),
        "model": str(meta.get("model", "dual_local_specialist")),
        "fallback": str(meta.get("fallback_strategy", "status_specialist")),
        "prompt_variant": "training_compatible",
        **evaluate_rows(eval_rows),
    }


def fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "category",
        "model",
        "fallback",
        "prompt_variant",
        "overall_acc",
        "coverage",
        "accuracy_on_decided",
        "wrong_rate",
        "abstention_rate",
        "avg_formal_runtime",
        "p95_formal_runtime",
        "avg_total_runtime",
        "p95_total_runtime",
        "correct",
        "wrong",
        "decided",
        "samples",
        "decision_sources",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "category",
        "model",
        "fallback",
        "prompt_variant",
        "overall_acc",
        "coverage",
        "wrong_rate",
        "avg_formal_runtime",
        "p95_formal_runtime",
        "avg_total_runtime",
        "p95_total_runtime",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join("---" for _ in fields) + "|\n")
        for row in rows:
            vals = []
            for field in fields:
                value = row.get(field, "")
                if field in {"overall_acc", "coverage", "wrong_rate"}:
                    vals.append(fmt_pct(value))
                elif field in {"avg_formal_runtime", "p95_formal_runtime", "avg_total_runtime", "p95_total_runtime"}:
                    vals.append(f"{float(value):.2f}s")
                else:
                    vals.append(str(value))
            f.write("| " + " | ".join(vals) + " |\n")


def pick_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep_methods = {
        "fixed10s",
        "fixed30s",
        "fixed60s",
        "fixed120s",
        "adaptive_anchor10",
        "adaptive_accuracy",
        "adaptive_accuracy120",
        "adaptive_correct60",
        "adaptive_correct120",
        "adaptive_expected_acc85",
        "adaptive_expected_acc90",
        "adaptive_expected_acc90_120",
        "adaptive_exact60",
        "model_only",
    }
    subset = [r for r in rows if r["method"] in keep_methods and is_maintrack_candidate(r)]
    # Keep formal-only baselines and the strongest row for each method/dataset.
    selected = []
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in subset:
        if row["category"] == "formal_only":
            selected.append(row)
        else:
            grouped.setdefault((row["dataset"], row["method"], row["prompt_variant"]), []).append(row)
    for group_rows in grouped.values():
        selected.append(max(group_rows, key=lambda r: (r["overall_acc"], -r["avg_total_runtime"], r["coverage"])))
    return sorted(selected, key=lambda r: (r["dataset"], r["category"], r["method"], r["prompt_variant"], -r["overall_acc"]))


def is_maintrack_candidate(row: Dict[str, Any]) -> bool:
    category = str(row.get("category"))
    if category == "hybrid_specialist":
        return False
    if category == "formal_only":
        return True
    if category == "model_only":
        return str(row.get("model")) == "qwen3_1_7b_100sft_cot" and str(row.get("fallback")) == "majority"
    if category == "hybrid":
        return (
            str(row.get("model")) == "qwen3_1_7b_100sft_cot"
            and str(row.get("fallback")) == "majority"
            and str(row.get("prompt_variant")) == "training_compatible"
        )
    return False


def best_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    allowed = [r for r in rows if is_maintrack_candidate(r)]
    out: Dict[str, Dict[str, Any]] = {}
    for split in sorted({r["dataset"] for r in allowed}):
        split_rows = [r for r in allowed if r["dataset"] == split]
        if split_rows:
            out[split] = max(split_rows, key=lambda r: (r["overall_acc"], -r["avg_formal_runtime"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--stage-strategies", default="fixed1s,fixed2s,fixed3s,fixed5s,fixed10s,fixed30s,fixed60s,fixed120s,adaptive_anchor10,adaptive_accuracy,adaptive_accuracy120,adaptive_correct60,adaptive_correct120,adaptive_expected_acc85,adaptive_expected_acc90,adaptive_expected_acc90_120,adaptive_balanced,adaptive_fast,adaptive_exact60")
    ap.add_argument("--fallback-log-suffixes", default="model_only,fallback,fallback_verieql_aware")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models = [str(m["key"]) for m in cfg["fallback"].get("models", []) if m.get("enabled", True)]
    stage_strategies = [x.strip() for x in args.stage_strategies.split(",") if x.strip()]
    suffixes = [x.strip() for x in args.fallback_log_suffixes.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for split in experiment_test_splits(cfg):
        for strategy in stage_strategies:
            stage_rows = load_stage(split, strategy)
            if not stage_rows:
                continue
            rows.append(evaluate_formal_only(split, strategy, stage_rows))
            for model_key in models:
                for suffix in suffixes:
                    if suffix == "model_only":
                        logs = load_model_logs(split, model_key)
                        prompt_variant = "training_compatible"
                    else:
                        logs = load_model_logs(split, model_key, strategy, suffix)
                        prompt_variant = suffix.replace("fallback_", "")
                    if not logs:
                        continue
                    for fallback_strategy in FALLBACK_STRATEGIES:
                        rows.append(evaluate_hybrid(split, strategy, stage_rows, logs, model_key, fallback_strategy, prompt_variant))
            for specialist_path in sorted((PROJECT_ROOT / "outputs/decisions").glob(f"{split}.{strategy}.dual_local_specialist.*.final.jsonl")):
                rows.append(evaluate_precomputed_final(split, strategy, specialist_path))
        for model_key in models:
            logs = load_model_logs(split, model_key)
            if not logs:
                continue
            for fallback_strategy in FALLBACK_STRATEGIES:
                rows.append(evaluate_model_only(split, logs, model_key, fallback_strategy))

    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    tables_dir = PROJECT_ROOT / "outputs/tables"
    write_json(metrics_dir / "optimized_experiment_suite.json", rows)
    write_csv(metrics_dir / "optimized_experiment_suite.csv", rows)
    write_markdown(tables_dir / "optimized_experiment_suite.md", rows)
    summary = pick_summary_rows(rows)
    write_markdown(tables_dir / "optimized_main_summary.md", summary)
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "best_by_dataset": best_by_dataset(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
