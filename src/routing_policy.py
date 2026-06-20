#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from common import PROJECT_ROOT, percentile, write_json
from extract_features import assign_bucket, features_one, flags_for_pair, merge_features


DETERMINED = {"equivalent", "non_equivalent"}
DEFAULT_POLICY_PATH = PROJECT_ROOT / "outputs/budget/routing_policy.json"


def determined_coverage(rows: List[Dict[str, Any]], budget: float) -> float:
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if r["normalized_status"] in DETERMINED and float(r["runtime_s"]) <= budget)
    return ok / len(rows)


def choose_budget(
    rows: List[Dict[str, Any]],
    budgets: List[int],
    epsilon: float,
    min_coverage: float,
    default_budget: int = 1,
    reference_budget: Optional[int] = None,
) -> int:
    if not rows:
        return default_budget
    budgets = sorted(set(int(x) for x in budgets))
    positive_budgets = [b for b in budgets if b > 0]
    if not positive_budgets:
        return 0
    ref_budget = int(reference_budget or max(budgets))
    reference_coverage = determined_coverage(rows, ref_budget)
    if reference_coverage < min_coverage:
        return 0
    for budget in positive_budgets:
        if reference_coverage - determined_coverage(rows, budget) <= epsilon:
            return int(budget)
    return int(max(positive_budgets))


def band(value: Any, cuts: List[int], labels: List[str]) -> str:
    try:
        v = int(value)
    except Exception:
        v = 0
    for cut, label in zip(cuts, labels):
        if v <= cut:
            return label
    return labels[-1]


def descriptor_from_standard_row(row: Dict[str, Any]) -> Dict[str, Any]:
    f1 = features_one(row["sql1"])
    f2 = features_one(row["sql2"])
    features = merge_features(f1, f2, row)
    static_flags, risk_flags = flags_for_pair(row["sql1"], row["sql2"])
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "bucket": assign_bucket(features),
        "features": features,
        "static_unsupported_flags": static_flags,
        "risk_flags": risk_flags,
    }


def descriptor_from_calibration_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket", "Unknown"),
        "features": row.get("features", {}),
        "static_unsupported_flags": row.get("static_unsupported_flags", []),
        "risk_flags": row.get("risk_flags", []),
    }


def feature_bins(descriptor: Dict[str, Any]) -> Dict[str, str]:
    features = descriptor.get("features", {})
    return {
        "bucket": str(descriptor.get("bucket", "Unknown")),
        "agg": "agg" if features.get("has_aggregation") or features.get("has_group_by") or features.get("has_having") else "noagg",
        "nested": band(features.get("nesting_depth", 0), [0, 1], ["nest0", "nest1", "nest2p"]),
        "set": "set" if features.get("has_set_op") else "noset",
        "join": band(features.get("join_count", 0), [0, 2], ["join0", "join1_2", "join3p"]),
        "tables": band(features.get("table_count", 0), [1, 3], ["table1", "tables2_3", "tables4p"]),
        "pred": band(features.get("predicate_count", 0), [4, 9], ["pred0_4", "pred5_9", "pred10p"]),
        "length": band(features.get("sql_length", 0), [49, 99, 199], ["len0_49", "len50_99", "len100_199", "len200p"]),
        "distinct": "distinct" if features.get("has_distinct") else "nodistinct",
        "null": "null" if features.get("has_null_predicate") else "nonull",
        "case": "case" if features.get("has_case_when") else "nocase",
        "static": "static" if descriptor.get("static_unsupported_flags") else "nostatic",
        "risk": "risk" if descriptor.get("risk_flags") else "norisk",
    }


def policy_key(descriptor: Dict[str, Any], level: str) -> str:
    b = feature_bins(descriptor)
    if level == "fine":
        parts = [
            level,
            b["bucket"],
            b["agg"],
            b["nested"],
            b["set"],
            b["join"],
            b["tables"],
            b["pred"],
            b["length"],
            b["distinct"],
            b["null"],
            b["case"],
            b["static"],
            b["risk"],
        ]
    elif level == "medium":
        parts = [
            level,
            b["bucket"],
            b["agg"],
            b["nested"],
            b["set"],
            b["join"],
            b["tables"],
            b["static"],
            b["risk"],
        ]
    elif level == "compat_bucket":
        parts = [level, b["bucket"], b["static"], b["risk"]]
    elif level == "bucket":
        parts = [level, b["bucket"]]
    else:
        raise ValueError(f"Unknown routing policy level: {level}")
    return "|".join(parts)


def summarize_group(
    name: str,
    rows: List[Dict[str, Any]],
    budgets: List[int],
    epsilon: float,
    min_coverage: float,
    default_budget: int,
    reference_budget: int,
) -> Dict[str, Any]:
    selected = choose_budget(rows, budgets, epsilon, min_coverage, default_budget, reference_budget)
    status_counts = Counter(str(row.get("normalized_status")) for row in rows)
    runtimes = [float(row["runtime_s"]) for row in rows if row.get("normalized_status") in DETERMINED]
    coverage_budgets = sorted(set([*budgets, reference_budget]))
    return {
        "key": name,
        "support": len(rows),
        "selected_budget": selected,
        "reference_budget": reference_budget,
        "reference_coverage": determined_coverage(rows, reference_budget),
        "coverage_by_budget": {str(b): determined_coverage(rows, b) for b in coverage_budgets},
        "status_counts": dict(status_counts),
        "determined_runtime_p50": percentile(runtimes, 50),
        "determined_runtime_p95": percentile(runtimes, 95),
    }


def learn_policy(
    rows: List[Dict[str, Any]],
    budgets: List[int],
    epsilon: float,
    min_coverage: float,
    default_budget: int,
    min_support_by_level: Dict[str, int],
    reference_budget: Optional[int] = None,
    zero_budget_levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ref_budget = int(reference_budget or max(budgets))
    levels = ["fine", "medium", "compat_bucket", "bucket"]
    if zero_budget_levels is None:
        zero_budget_levels = ["fine", "medium", "compat_bucket"]
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {level: defaultdict(list) for level in levels}
    for row in rows:
        descriptor = descriptor_from_calibration_row(row)
        for level in levels:
            grouped[level][policy_key(descriptor, level)].append(row)

    policy_levels = []
    for level in levels:
        min_support = int(min_support_by_level.get(level, 1))
        groups = {}
        for key, part in grouped[level].items():
            if len(part) < min_support:
                continue
            groups[key] = summarize_group(key, part, budgets, epsilon, min_coverage, default_budget, ref_budget)
        policy_levels.append({"name": level, "min_support": min_support, "groups": groups})

    bucket_table = {
        key.replace("bucket|", "", 1): value["selected_budget"]
        for key, value in policy_levels[-1]["groups"].items()
        if key.startswith("bucket|")
    }
    return {
        "version": "profile_calibrated_selective_zero_budget_v2",
        "description": "Hierarchical routing policy learned from precomputed VeriEQL runtime profiles. Zero-budget bypass is only trusted at configured fine-grained levels; coarse fallbacks preserve a short VeriEQL probe.",
        "budgets_sec": budgets,
        "reference_budget_sec": ref_budget,
        "max_policy_budget_sec": max(budgets),
        "epsilon": epsilon,
        "min_coverage_threshold": min_coverage,
        "default_budget_sec": default_budget,
        "zero_budget_levels": zero_budget_levels,
        "fallback_order": levels,
        "bucket_budget_table": bucket_table,
        "levels": policy_levels,
    }


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_policy(path: str | Path, policy: Dict[str, Any]) -> None:
    write_json(path, policy)


def select_route(
    descriptor: Dict[str, Any],
    policy: Dict[str, Any],
    default_budget: int = 1,
) -> Dict[str, Any]:
    level_index = {level["name"]: level for level in policy.get("levels", [])}
    zero_budget_levels = set(policy.get("zero_budget_levels", ["fine", "medium", "compat_bucket"]))
    for level in policy.get("fallback_order", ["fine", "medium", "bucket"]):
        spec = level_index.get(level)
        if not spec:
            continue
        key = policy_key(descriptor, level)
        group = spec.get("groups", {}).get(key)
        if not group:
            continue
        selected = int(group.get("selected_budget", default_budget))
        if selected <= 0:
            if level not in zero_budget_levels:
                fallback_budget = int(policy.get("default_budget_sec", default_budget))
                return {
                    "route": "run_verieql" if fallback_budget > 0 else "zero_budget",
                    "selected_budget": max(0, fallback_budget),
                    "policy_level": level,
                    "policy_key": key,
                    "policy_support": group.get("support"),
                    "policy_reference_coverage": group.get("reference_coverage", 0.0),
                    "route_reason": f"coarse_low_coverage_guard_default_budget={fallback_budget}",
                }
            return {
                "route": "zero_budget",
                "selected_budget": 0,
                "policy_level": level,
                "policy_key": key,
                "policy_support": group.get("support"),
                "policy_reference_coverage": group.get("reference_coverage", 0.0),
                "route_reason": f"reference_coverage={group.get('reference_coverage', 0.0):.4f}",
            }
        return {
            "route": "run_verieql",
            "selected_budget": selected,
            "policy_level": level,
            "policy_key": key,
            "policy_support": group.get("support"),
            "policy_reference_coverage": group.get("reference_coverage", 0.0),
            "route_reason": f"selected_budget={selected}",
        }

    fallback_budget = int(policy.get("default_budget_sec", default_budget))
    return {
        "route": "run_verieql" if fallback_budget > 0 else "zero_budget",
        "selected_budget": max(0, fallback_budget),
        "policy_level": "default",
        "policy_key": "default",
        "policy_support": None,
        "policy_reference_coverage": None,
        "route_reason": "no_matching_policy_group",
    }


def route_rows(rows: Iterable[Dict[str, Any]], policy: Dict[str, Any], default_budget: int = 1) -> List[Dict[str, Any]]:
    routed = []
    for row in rows:
        descriptor = descriptor_from_standard_row(row)
        route = select_route(descriptor, policy, default_budget)
        routed.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "gold_label": row.get("label"),
                **descriptor,
                **route,
            }
        )
    return routed


def summarize_routes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_route = Counter(row["route"] for row in rows)
    by_budget = Counter(str(row["selected_budget"]) for row in rows)
    by_level = Counter(str(row.get("policy_level")) for row in rows)
    by_bucket_route: Dict[str, Counter[str]] = defaultdict(Counter)
    by_static_flag = Counter(flag for row in rows for flag in row.get("static_unsupported_flags", []))
    zero_static = sum(1 for row in rows if row["route"] == "zero_budget" and row.get("static_unsupported_flags"))
    for row in rows:
        by_bucket_route[str(row.get("bucket"))][row["route"]] += 1
    return {
        "samples": len(rows),
        "by_route": dict(by_route),
        "by_budget": dict(by_budget),
        "by_policy_level": dict(by_level),
        "by_bucket_route": {bucket: dict(counter) for bucket, counter in sorted(by_bucket_route.items())},
        "static_flag_counts": dict(by_static_flag),
        "zero_budget_with_static_flags": zero_static,
    }
