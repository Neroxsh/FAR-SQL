#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, mean, percentile, read_jsonl, update_manifest, write_json, write_jsonl


DETERMINED = {"equivalent", "non_equivalent"}
CHECKED_PREFIX_STATES = {"EQU", "UNSAT", "CHECKED", "VERIFIED"}


def majority_label(log: Optional[Dict[str, Any]]) -> Optional[str]:
    if not log:
        return None
    parsed = [x for x in log.get("parsed_answers", []) if x in {"yes", "no"}]
    if not parsed:
        return None
    counts = Counter(parsed)
    if counts["yes"] == counts["no"]:
        return None
    return "yes" if counts["yes"] > counts["no"] else "no"


def verieql_label(status: str) -> Optional[str]:
    if status == "equivalent":
        return "yes"
    if status == "non_equivalent":
        return "no"
    return None


def is_timeout_positive_prior(stage: Dict[str, Any]) -> bool:
    if stage.get("verieql_status") != "timeout":
        return False
    risk_flags = set(str(x) for x in stage.get("risk_flags", []))
    if "correlated_or_subquery" not in risk_flags:
        return False
    observed = [str(x).upper() for x in stage.get("observed_states") or stage.get("raw_states") or []]
    prefix = [x for x in observed if x != "TMO"]
    return bool(prefix) and all(x in CHECKED_PREFIX_STATES for x in prefix)


def load_logs(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def choose_route(stage: Dict[str, Any]) -> str:
    status = str(stage.get("verieql_status"))
    if is_timeout_positive_prior(stage):
        return "timeout_positive_prior"
    if status == "runtime_error":
        return "trace_guided_runtime_error"
    if status == "conversion_error":
        return "witness_guided_conversion_error"
    return "base_majority"


def make_final(
    stage: Dict[str, Any],
    base_logs: Dict[str, Dict[str, Any]],
    trace_logs: Dict[str, Dict[str, Any]],
    witness_logs: Dict[str, Dict[str, Any]],
    unified_logs: Optional[Dict[str, Dict[str, Any]]],
    model_key: str,
    use_timeout_aggregation_witness: bool,
    timeout_positive_prior_mode: str,
    unified_prompt_variant: Optional[str],
) -> Dict[str, Any]:
    sid = str(stage["id"])
    vlabel = verieql_label(str(stage["verieql_status"]))
    fallback_runtime = 0.0
    router_route = "verified"
    router_prompt_variant = None

    if vlabel is not None:
        final_label = vlabel
        decision_source = "verified"
        fallback_outputs: List[str] = []
        fallback_parsed: List[str] = []
    else:
        router_route = choose_route(stage)
        if router_route == "timeout_positive_prior":
            if timeout_positive_prior_mode == "direct_yes":
                final_label = "yes"
                decision_source = "trace_prior"
                fallback_outputs = []
                fallback_parsed = []
            else:
                log = base_logs.get(sid)
                router_route = "timeout_positive_prior_llm"
                router_prompt_variant = "training_compatible"
                final_label = majority_label(log)
                fallback_runtime = float(log.get("fallback_runtime", 0.0) or 0.0) if log else 0.0
                fallback_outputs = list(log.get("raw_outputs", [])) if log else []
                fallback_parsed = list(log.get("parsed_answers", [])) if log else []
                decision_source = "model_assisted" if final_label in {"yes", "no"} else "abstained"
                if final_label is None:
                    final_label = "uncertain"
        else:
            if unified_logs is not None:
                log = unified_logs.get(sid)
                router_prompt_variant = unified_prompt_variant or "training_with_verifier_observation"
            elif use_timeout_aggregation_witness and stage.get("verieql_status") == "timeout" and stage.get("bucket") == "Aggregation":
                log = witness_logs.get(sid)
                router_route = "witness_guided_timeout_aggregation"
                router_prompt_variant = "witness_guided"
            elif router_route == "trace_guided_runtime_error":
                log = trace_logs.get(sid)
                router_prompt_variant = "trace_guided"
            elif router_route == "witness_guided_conversion_error":
                log = witness_logs.get(sid)
                router_prompt_variant = "witness_guided"
            else:
                log = base_logs.get(sid)
                router_prompt_variant = "training_compatible"
            final_label = majority_label(log)
            fallback_runtime = float(log.get("fallback_runtime", 0.0) or 0.0) if log else 0.0
            fallback_outputs = list(log.get("raw_outputs", [])) if log else []
            fallback_parsed = list(log.get("parsed_answers", [])) if log else []
            decision_source = "model_assisted" if final_label in {"yes", "no"} else "abstained"
            if final_label is None:
                final_label = "uncertain"

    formal_runtime = float(stage.get("verieql_runtime", 0.0) or 0.0)
    is_decided = final_label in {"yes", "no"}
    is_correct = (final_label == stage["gold_label"]) if is_decided else None
    return {
        "id": sid,
        "dataset": stage["dataset"],
        "method": "adaptive_traceprior_router",
        "category": "hybrid_status_aware_router",
        "model": model_key,
        "fallback_strategy": "majority",
        "gold_label": stage["gold_label"],
        "bucket": stage.get("bucket"),
        "selected_budget": stage.get("selected_budget"),
        "verieql_status": stage["verieql_status"],
        "verieql_runtime": formal_runtime,
        "router_route": router_route,
        "router_prompt_variant": router_prompt_variant,
        "fallback_outputs": fallback_outputs,
        "fallback_parsed": fallback_parsed,
        "final_label": final_label,
        "decision_source": decision_source,
        "total_runtime": formal_runtime + fallback_runtime,
        "is_decided": is_decided,
        "is_correct": is_correct,
        "is_wrong_decision": bool(is_decided and not is_correct),
        "risk_flags": stage.get("risk_flags", []),
        "static_unsupported_flags": stage.get("static_unsupported_flags", []),
    }


def evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    decided = [r for r in rows if r["is_decided"]]
    correct = [r for r in rows if r["is_correct"] is True]
    wrong = [r for r in rows if r["is_decided"] and r["is_correct"] is False]
    formal_runtimes = [float(r.get("verieql_runtime", 0.0) or 0.0) for r in rows]
    total_runtimes = [float(r.get("total_runtime", 0.0) or 0.0) for r in rows]
    return {
        "samples": len(rows),
        "correct": len(correct),
        "decided": len(decided),
        "wrong": len(wrong),
        "overall_acc": len(correct) / len(rows) if rows else 0.0,
        "coverage": len(decided) / len(rows) if rows else 0.0,
        "accuracy_on_decided": len(correct) / len(decided) if decided else 0.0,
        "avg_formal_runtime": mean(formal_runtimes),
        "p95_formal_runtime": percentile(formal_runtimes, 95),
        "avg_total_runtime": mean(total_runtimes),
        "p95_total_runtime": percentile(total_runtimes, 95),
        "decision_sources": dict(Counter(str(r.get("decision_source")) for r in rows)),
        "router_routes": dict(Counter(str(r.get("router_route")) for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--strategy", default="adaptive_accuracy")
    ap.add_argument("--model-key", default="qwen3_1_7b_100sft_cot")
    ap.add_argument("--output-tag", default="status_aware_router")
    ap.add_argument("--use-timeout-aggregation-witness", action="store_true")
    ap.add_argument(
        "--timeout-positive-prior-mode",
        choices=["direct_yes", "llm_verify"],
        default="llm_verify",
        help=(
            "Use llm_verify for the strict FAR-SQL profile. "
            "Use direct_yes only to reproduce the submitted paper artifacts."
        ),
    )
    ap.add_argument("--unified-log-suffix", default=None)
    ap.add_argument("--unified-prompt-variant", default="training_with_verifier_observation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs: Dict[str, Any] = {}
    metrics_rows: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    for split in experiment_test_splits(cfg):
        stage_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{args.strategy}.verieql_stage.jsonl"
        stage_rows = read_jsonl(stage_path)
        base_logs = load_logs(PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{args.model_key}.model_only.jsonl")
        trace_logs = load_logs(PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{args.strategy}.{args.model_key}.fallback_trace_guided.jsonl")
        witness_logs = load_logs(PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{args.strategy}.{args.model_key}.fallback_witness_guided.jsonl")
        unified_logs = (
            load_logs(PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{args.strategy}.{args.model_key}.{args.unified_log_suffix}.jsonl")
            if args.unified_log_suffix
            else None
        )

        final_rows = [
            make_final(
                stage,
                base_logs,
                trace_logs,
                witness_logs,
                unified_logs,
                args.model_key,
                args.use_timeout_aggregation_witness,
                args.timeout_positive_prior_mode,
                args.unified_prompt_variant,
            )
            for stage in stage_rows
        ]
        out_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{args.strategy}.{args.model_key}.{args.output_tag}.final.jsonl"
        outputs[split] = {"path": str(out_path), "rows": write_jsonl(out_path, final_rows)}
        metrics_rows[split] = evaluate_rows(final_rows)
        all_rows.extend(final_rows)

    metrics_rows["combined"] = evaluate_rows(all_rows)
    metrics_path = PROJECT_ROOT / "outputs/metrics" / f"{args.output_tag}.metrics.json"
    write_json(metrics_path, metrics_rows)
    outputs["metrics"] = {"path": str(metrics_path)}
    update_manifest(PROJECT_ROOT / "outputs", "run_status_aware_router", outputs)
    print(json.dumps({"outputs": outputs, "metrics": metrics_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
