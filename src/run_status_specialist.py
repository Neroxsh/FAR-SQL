#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import PROJECT_ROOT, experiment_test_splits, load_config, mean, percentile, read_jsonl, update_manifest, write_json, write_jsonl
from fuse_decisions import label_from_verieql_status


PRIMARY_MODEL_KEY = "qwen3_1_7b_100sft_cot"
SPECIALIST_MODEL_KEY = "qwen25_coder_1_5b_100sft_cot"
DEFAULT_ROUTE_RULE = {
    "timeout": "q3_majority",
    "conversion_error": "q25_direct",
    "unsupported_runtime": "q25_direct",
    "runtime_error": "q3_direct",
    "zero_budget_routed": "q3_majority",
}


def load_logs(split: str, model_key: str) -> Dict[str, Dict[str, Any]]:
    path = PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl"
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def majority_label(log: Optional[Dict[str, Any]]) -> Optional[str]:
    if not log:
        return None
    valid = [x for x in log.get("parsed_answers", []) if x in {"yes", "no"}]
    if not valid:
        return None
    counts = Counter(valid)
    if counts["yes"] == counts["no"]:
        return None
    return "yes" if counts["yes"] > counts["no"] else "no"


def direct_label(log: Optional[Dict[str, Any]]) -> Optional[str]:
    if not log:
        return None
    parsed = list(log.get("parsed_answers", []))
    return parsed[0] if parsed and parsed[0] in {"yes", "no"} else None


def runtime_of(log: Optional[Dict[str, Any]]) -> float:
    return float(log.get("fallback_runtime", 0.0) or 0.0) if log else 0.0


def select_prediction(
    status: str,
    primary_log: Optional[Dict[str, Any]],
    specialist_log: Optional[Dict[str, Any]],
    route_rule: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], str, float]:
    route = route_rule.get(status, "q3_majority")
    primary_majority = majority_label(primary_log)
    primary_direct = direct_label(primary_log)
    specialist_majority = majority_label(specialist_log)
    specialist_direct = direct_label(specialist_log)
    route_options = {
        "q3_majority": (primary_majority, PRIMARY_MODEL_KEY, "model_assisted_primary", runtime_of(primary_log)),
        "q3_direct": (primary_direct, PRIMARY_MODEL_KEY, "model_assisted_primary_direct", runtime_of(primary_log)),
        "q25_majority": (specialist_majority, SPECIALIST_MODEL_KEY, "model_assisted_specialist_majority", runtime_of(specialist_log)),
        "q25_direct": (specialist_direct, SPECIALIST_MODEL_KEY, "model_assisted_specialist", runtime_of(specialist_log)),
    }
    label, model_key, decision_source, fallback_runtime = route_options.get(route, route_options["q3_majority"])
    return label, model_key, decision_source, route, fallback_runtime


def build_rows(
    stage_rows: List[Dict[str, Any]],
    primary_logs: Dict[str, Dict[str, Any]],
    specialist_logs: Dict[str, Dict[str, Any]],
    route_rule: Dict[str, str],
    router_name: str,
) -> List[Dict[str, Any]]:
    rows = []
    for stage in stage_rows:
        verified = label_from_verieql_status(str(stage.get("verieql_status")))
        if verified in {"yes", "no"}:
            final_label = verified
            decision_source = "verified"
            selected_fallback = None
            selected_route = "verified"
            fallback_runtime = 0.0
        else:
            primary_log = primary_logs.get(str(stage["id"]))
            specialist_log = specialist_logs.get(str(stage["id"]))
            status = str(stage.get("verieql_status"))
            selected_route = route_rule.get(status, "q3_majority")
            selected, selected_fallback, decision_source, selected_route, fallback_runtime = select_prediction(
                status, primary_log, specialist_log, route_rule
            )
            if selected in {"yes", "no"}:
                final_label = selected
            else:
                final_label = "uncertain"
                decision_source = "abstained"
                selected_fallback = None
                selected_route = f"{selected_route}:abstained"
                fallback_runtime = max(runtime_of(primary_log), runtime_of(specialist_log))
        is_decided = final_label in {"yes", "no"}
        rows.append(
            {
                "id": stage["id"],
                "dataset": stage["dataset"],
                "gold_label": stage["gold_label"],
                "bucket": stage.get("bucket"),
                "selected_budget": stage.get("selected_budget"),
                "verieql_status": stage.get("verieql_status"),
                "verieql_label": stage.get("verieql_label"),
                "verieql_runtime": float(stage.get("verieql_runtime", 0.0) or 0.0),
                "category": "hybrid_specialist",
                "model": "dual_local_specialist",
                "fallback_strategy": router_name,
                "fallback_model_key": selected_fallback,
                "router_rule": route_rule,
                "selected_route": selected_route,
                "final_label": final_label,
                "decision_source": decision_source,
                "total_runtime": float(stage.get("verieql_runtime", 0.0) or 0.0) + fallback_runtime,
                "is_decided": is_decided,
                "is_correct": (final_label == stage["gold_label"]) if is_decided else None,
                "is_wrong_decision": bool(is_decided and final_label != stage["gold_label"]),
                "risk_flags": stage.get("risk_flags", []),
                "static_unsupported_flags": stage.get("static_unsupported_flags", []),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    decided = [r for r in rows if r["is_decided"]]
    correct = [r for r in rows if r["is_correct"] is True]
    wrong = [r for r in rows if r["is_decided"] and r["is_correct"] is False]
    formal = [float(r.get("verieql_runtime", 0.0) or 0.0) for r in rows]
    total = [float(r.get("total_runtime", 0.0) or 0.0) for r in rows]
    return {
        "samples": n,
        "overall_acc": len(correct) / n if n else 0.0,
        "coverage": len(decided) / n if n else 0.0,
        "wrong_rate": len(wrong) / n if n else 0.0,
        "avg_formal_runtime": mean(formal),
        "p95_formal_runtime": percentile(formal, 95),
        "avg_total_runtime": mean(total),
        "p95_total_runtime": percentile(total, 95),
        "decision_sources": dict(Counter(str(r.get("decision_source")) for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--strategies", default="fixed10s,fixed30s,adaptive_accuracy")
    ap.add_argument(
        "--router-name",
        default="failure_mode_router",
        help="Name recorded in the final decision files and summary tables.",
    )
    ap.add_argument(
        "--route-rule",
        default="timeout=q3_majority,conversion_error=q25_direct,unsupported_runtime=q25_direct,runtime_error=q3_direct,zero_budget_routed=q3_majority",
        help="Comma-separated verieql_status=model_strategy mapping.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    route_rule = dict(DEFAULT_ROUTE_RULE)
    for item in [x.strip() for x in args.route_rule.split(",") if x.strip()]:
        key, value = item.split("=", 1)
        route_rule[key.strip()] = value.strip()

    outputs: Dict[str, Any] = {"decisions": {}, "metrics": {}}
    metric_rows: List[Dict[str, Any]] = []
    for split in experiment_test_splits(cfg):
        primary_logs = load_logs(split, PRIMARY_MODEL_KEY)
        specialist_logs = load_logs(split, SPECIALIST_MODEL_KEY)
        for strategy in strategies:
            stage_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.verieql_stage.jsonl"
            if not stage_path.exists():
                continue
            stage_rows = read_jsonl(stage_path)
            final_rows = build_rows(stage_rows, primary_logs, specialist_logs, route_rule, args.router_name)
            out_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{strategy}.dual_local_specialist.{args.router_name}.final.jsonl"
            write_jsonl(out_path, final_rows)
            outputs["decisions"][f"{split}:{strategy}"] = str(out_path)
            metric_rows.append(
                {
                    "dataset": split,
                    "method": strategy,
                    "category": "hybrid_specialist",
                    "model": "dual_local_specialist",
                    "fallback": args.router_name,
                    "prompt_variant": "training_compatible",
                    **summarize(final_rows),
                }
            )

    metrics_path = PROJECT_ROOT / "outputs/metrics" / "status_specialist_metrics.json"
    write_json(metrics_path, metric_rows)
    outputs["metrics"]["json"] = str(metrics_path)
    update_manifest(PROJECT_ROOT / "outputs", "run_status_specialist", outputs)
    print(json.dumps({"rows": len(metric_rows), "metrics_path": str(metrics_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
