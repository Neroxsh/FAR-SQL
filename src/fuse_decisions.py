#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PROJECT_ROOT, experiment_test_splits, load_config, read_jsonl, update_manifest, write_jsonl


DETERMINED = {"equivalent", "non_equivalent"}
UNCERTAIN_STATUS_FOR_RISK_GATE = {"unknown", "conversion_error", "runtime_error"}
FRAGILE_VERIEQL_FLAGS = {"union_all", "limit_offset", "exists"}


def label_from_verieql_status(status: str) -> Optional[str]:
    if status == "equivalent":
        return "yes"
    if status == "non_equivalent":
        return "no"
    return None


def fallback_label(log: Optional[Dict[str, Any]], strategy: str, stage: Dict[str, Any]) -> Optional[str]:
    if log is None:
        return None
    parsed = [x for x in log.get("parsed_answers", [])]
    valid = [x for x in parsed if x in {"yes", "no"}]
    if strategy == "always_output":
        return parsed[0] if parsed and parsed[0] in {"yes", "no"} else None
    if strategy == "majority":
        if not valid:
            return None
        counts = Counter(valid)
        if counts["yes"] == counts["no"]:
            return None
        return "yes" if counts["yes"] > counts["no"] else "no"
    if len(valid) != len(parsed) or not valid:
        return None
    if len(set(valid)) != 1:
        return None
    agreed = valid[0]
    if strategy == "risk_gated":
        high_risk = bool(stage.get("risk_flags")) or bool(stage.get("static_unsupported_flags"))
        if high_risk and stage.get("verieql_status") in UNCERTAIN_STATUS_FOR_RISK_GATE:
            return None
    return agreed


def make_final(stage: Dict[str, Any], log: Optional[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    vlabel = label_from_verieql_status(stage["verieql_status"])
    if vlabel is not None:
        final_label = vlabel
        decision_source = "verified"
        fallback_outputs: List[Any] = []
        fallback_parsed: List[Any] = []
        fallback_runtime = 0.0
    else:
        flabel = fallback_label(log, strategy, stage)
        fallback_outputs = log.get("raw_outputs", []) if log else []
        fallback_parsed = log.get("parsed_answers", []) if log else []
        fallback_runtime = float(log.get("fallback_runtime", 0.0)) if log else 0.0
        if flabel is None:
            final_label = "uncertain"
            decision_source = "abstained"
        else:
            final_label = flabel
            decision_source = "model_assisted"
    is_decided = final_label in {"yes", "no"}
    is_correct = (final_label == stage["gold_label"]) if is_decided else None
    return {
        "id": stage["id"],
        "dataset": stage["dataset"],
        "gold_label": stage["gold_label"],
        "bucket": stage.get("bucket"),
        "selected_budget": stage.get("selected_budget"),
        "verieql_status": stage["verieql_status"],
        "verieql_label": stage.get("verieql_label"),
        "verieql_runtime": stage.get("verieql_runtime", 0.0),
        "fallback_triggered": stage.get("fallback_triggered", False),
        "fallback_outputs": fallback_outputs,
        "fallback_parsed": fallback_parsed,
        "fallback_strategy": strategy,
        "fallback_model_key": log.get("model_key") if log else None,
        "final_label": final_label,
        "decision_source": decision_source,
        "total_runtime": float(stage.get("verieql_runtime", 0.0)) + fallback_runtime,
        "is_decided": is_decided,
        "is_correct": is_correct,
        "is_wrong_decision": bool(is_decided and not is_correct),
        "risk_flags": stage.get("risk_flags", []),
        "static_unsupported_flags": stage.get("static_unsupported_flags", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--verieql-strategy", default=None)
    ap.add_argument("--log-suffix", default="fallback")
    ap.add_argument("--output-suffix", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.verieql_strategy is None:
        args.verieql_strategy = str(cfg.get("experiments", {}).get("autobudget_strategy_name", "autobudget"))
    strategies = cfg["fallback"]["strategies"]
    models = [m for m in cfg["fallback"].get("models", []) if m.get("enabled", True)]
    outputs = {}
    for split in experiment_test_splits(cfg):
        stage_rows = read_jsonl(PROJECT_ROOT / "outputs/decisions" / f"{split}.{args.verieql_strategy}.verieql_stage.jsonl")
        for model in models:
            model_key = str(model["key"])
            log_path = PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.{args.verieql_strategy}.{model_key}.{args.log_suffix}.jsonl"
            log_source = "strategy_fallback"
            if not log_path.exists():
                log_path = PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl"
                log_source = "model_only_reuse"
            if not log_path.exists():
                continue
            logs = {row["id"]: row for row in read_jsonl(log_path)}
            for strategy in strategies:
                final_rows = [make_final(row, logs.get(row["id"]), strategy) for row in stage_rows]
                for row in final_rows:
                    row["fallback_log_source"] = log_source
                suffix = f".{args.output_suffix}" if args.output_suffix else ""
                out_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{args.verieql_strategy}.{model_key}.{strategy}{suffix}.final.jsonl"
                outputs[f"{split}:{model_key}:{strategy}"] = {"path": str(out_path), "rows": write_jsonl(out_path, final_rows)}
    update_manifest(PROJECT_ROOT / "outputs", "fuse_decisions", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
