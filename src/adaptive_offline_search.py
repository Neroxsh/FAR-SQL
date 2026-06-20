#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import PROJECT_ROOT, load_config, mean, percentile, read_jsonl, sql_pair_hash, standard_paths, update_manifest, write_json, write_jsonl
from extract_features import assign_bucket, features_one, flags_for_pair, merge_features
from routing_policy import descriptor_from_standard_row
from verieql_simulation import DETERMINED, simulate_verieql_record


AVAILABLE_TEST_BUDGETS = [0, 1, 2, 3, 5, 10, 30, 60, 120]
POSITIVE_TEST_BUDGETS = [b for b in AVAILABLE_TEST_BUDGETS if b > 0]
REFUTED_STATES = {"NEQ", "SAT", "REFUTED"}
CHECKED_STATES = {"EQU", "UNSAT", "CHECKED", "VERIFIED"}
ZERO_SOURCE_FLAGS = {
    "window_over",
    "exists",
    "select_subquery",
    "unsupported_statistical_agg",
    "advanced_grouping_or_filter",
    "select_without_from",
    "values_clause",
}
ESCALATE10_RISK_FLAGS = {"correlated_or_subquery", "union_all"}
UNSUPPORTEDISH = {"unsupported_static", "unsupported_runtime", "conversion_error", "runtime_error"}


VARIANTS = {
    "adaptive_exact60": {
        "description": "Accuracy-preserving profile compression: choose the smallest budget that matches the 60s calibration coverage for each profile.",
        "objective": "coverage",
        "target_budget": 60,
        "epsilon": 0.0,
        "long_target_budget": 60,
        "long_gain_threshold": 1.0,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_anchor10": {
        "description": "Anchor to Fixed@10; reduce budgets only when profile coverage is almost unchanged.",
        "objective": "coverage",
        "target_budget": 10,
        "epsilon": 0.002,
        "long_target_budget": 10,
        "long_gain_threshold": 1.0,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_accuracy": {
        "description": "Maximize expected formal coverage; allow longer budgets when profile says they recover decisions.",
        "objective": "coverage",
        "target_budget": 60,
        "epsilon": 0.005,
        "long_gain_threshold": 0.005,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_accuracy120": {
        "description": "Maximize expected formal coverage with an optional 120s long-tail action for profiles that still gain beyond 60s.",
        "objective": "coverage",
        "target_budget": 120,
        "epsilon": 0.005,
        "long_gain_threshold": 0.005,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_correct60": {
        "description": "Label-aware compression: choose the smallest budget that preserves 60s formal correct-coverage on the calibration profile.",
        "objective": "correct_coverage",
        "target_budget": 60,
        "epsilon": 0.002,
        "long_target_budget": 60,
        "long_gain_threshold": 1.0,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_correct120": {
        "description": "Label-aware compression: choose the smallest budget that preserves 120s formal correct-coverage on the calibration profile.",
        "objective": "correct_coverage",
        "target_budget": 120,
        "epsilon": 0.002,
        "long_target_budget": 120,
        "long_gain_threshold": 1.0,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_expected_acc85": {
        "description": "Expected-accuracy cascade: choose the fastest budget whose estimated hybrid accuracy is within tolerance of the best profile budget, assuming an 85% fallback prior.",
        "objective": "hybrid_utility",
        "target_budget": 60,
        "fallback_prior": 0.85,
        "epsilon": 0.003,
        "latency_weight": 0.002,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_expected_acc90": {
        "description": "Expected-accuracy cascade with a stronger fallback prior; useful when the local SFT model is empirically reliable on unresolved cases.",
        "objective": "hybrid_utility",
        "target_budget": 60,
        "fallback_prior": 0.90,
        "epsilon": 0.003,
        "latency_weight": 0.002,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_expected_acc90_120": {
        "description": "Expected-accuracy cascade with a 120s long-tail action; spend extra symbolic time only when profile utility still beats early fallback.",
        "objective": "hybrid_utility",
        "target_budget": 120,
        "fallback_prior": 0.90,
        "epsilon": 0.003,
        "latency_weight": 0.002,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_balanced": {
        "description": "Keep Fixed@10 coverage within a small tolerance, using long budgets only for clear gains.",
        "objective": "coverage",
        "target_budget": 10,
        "epsilon": 0.010,
        "long_target_budget": 60,
        "long_gain_threshold": 0.030,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
    "adaptive_fast": {
        "description": "Aggressively compress budgets while keeping most Fixed@10 profile coverage.",
        "objective": "coverage",
        "target_budget": 10,
        "epsilon": 0.025,
        "long_target_budget": 30,
        "long_gain_threshold": 0.080,
        "min_support": {"fine": 80, "medium": 160, "coarse": 80, "bucket": 1},
    },
}


def band(value: Any, cuts: List[int], labels: List[str]) -> str:
    try:
        v = int(value)
    except Exception:
        v = 0
    for cut, label in zip(cuts, labels):
        if v <= cut:
            return label
    return labels[-1]


def descriptor_from_raw(row: Dict[str, Any], row_id: str) -> Optional[Dict[str, Any]]:
    pair = row.get("pair")
    if not isinstance(pair, list) or len(pair) < 2 or not isinstance(pair[0], str) or not isinstance(pair[1], str):
        return None
    std = {
        "id": row_id,
        "dataset": "calibration",
        "schema": row.get("schema", {}),
        "constraint": row.get("constraint"),
        "sql1": pair[0],
        "sql2": pair[1],
    }
    f1 = features_one(std["sql1"])
    f2 = features_one(std["sql2"])
    features = merge_features(f1, f2, std)
    static_flags, risk_flags = flags_for_pair(std["sql1"], std["sql2"])
    return {
        "id": row_id,
        "dataset": "calibration",
        "features": features,
        "bucket": assign_bucket(features),
        "static_unsupported_flags": static_flags,
        "risk_flags": risk_flags,
        "source_file": row.get("file"),
        "source_index": row.get("index"),
    }


def feature_bins(descriptor: Dict[str, Any]) -> Dict[str, str]:
    f = descriptor.get("features", {})
    flags = sorted(descriptor.get("static_unsupported_flags") or [])
    risk = sorted(descriptor.get("risk_flags") or [])
    return {
        "bucket": str(descriptor.get("bucket", "Unknown")),
        "agg": "agg" if f.get("has_aggregation") or f.get("has_group_by") or f.get("has_having") else "noagg",
        "nested": band(f.get("nesting_depth", 0), [0, 1], ["nest0", "nest1", "nest2p"]),
        "set": "set" if f.get("has_set_op") else "noset",
        "join": band(f.get("join_count", 0), [0, 2], ["join0", "join1_2", "join3p"]),
        "tables": band(f.get("table_count", 0), [1, 3], ["table1", "tables2_3", "tables4p"]),
        "pred": band(f.get("predicate_count", 0), [4, 9], ["pred0_4", "pred5_9", "pred10p"]),
        "length": band(f.get("sql_length", 0), [49, 99, 199], ["len0_49", "len50_99", "len100_199", "len200p"]),
        "distinct": "distinct" if f.get("has_distinct") else "nodistinct",
        "null": "null" if f.get("has_null_predicate") else "nonull",
        "case": "case" if f.get("has_case_when") else "nocase",
        "static": "+".join(flags) if flags else "nostatic",
        "risk": "+".join(risk) if risk else "norisk",
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
        parts = [level, b["bucket"], b["agg"], b["nested"], b["set"], b["join"], b["tables"], b["static"]]
    elif level == "coarse":
        parts = [level, b["bucket"], b["static"], "risk" if b["risk"] != "norisk" else "norisk"]
    elif level == "bucket":
        parts = [level, b["bucket"]]
    else:
        raise ValueError(f"Unknown level: {level}")
    return "|".join(parts)


def load_test_hashes(cfg: Dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for path in standard_paths(cfg).values():
        if not path.exists():
            continue
        for row in read_jsonl(path):
            hashes.add(sql_pair_hash(row["sql1"], row["sql2"], row.get("schema")))
    return hashes


def calibration_label_from_states(raw: Dict[str, Any]) -> Optional[str]:
    """Recover the public LeetCode label with the repository split script rule.

    This mirrors the LeetCode split rule used for the calibration source: any
    refutation token makes the pair non-equivalent; otherwise any checked token
    makes it equivalent; undecided records remain unlabeled calibration examples
    for coverage-only statistics.
    """
    states = [str(x).strip().upper() for x in (raw.get("states") or []) if x is not None]
    if any(state in REFUTED_STATES for state in states):
        return "no"
    if any(state in CHECKED_STATES for state in states):
        return "yes"
    return None


def load_calibration_rows(cfg: Dict[str, Any], budgets: List[int], limit: Optional[int]) -> List[Dict[str, Any]]:
    source = Path(cfg["data_sources"]["calibration"]["path"])
    sample_bound = int(cfg["verieql"]["sample_bound"])
    excluded = load_test_hashes(cfg)
    rows = []
    skipped_overlap = 0
    for row_no, raw in enumerate(read_jsonl(source), 1):
        pair = raw.get("pair")
        if isinstance(pair, list) and len(pair) >= 2:
            if sql_pair_hash(pair[0], pair[1], raw.get("schema", {})) in excluded:
                skipped_overlap += 1
                continue
        descriptor = descriptor_from_raw(raw, f"calibration:{row_no}")
        if descriptor is None:
            continue
        sims = {str(b): simulate_verieql_record(raw, b, sample_bound) for b in budgets if b > 0}
        rows.append(
            {
                **descriptor,
                "gold_label": calibration_label_from_states(raw),
                "simulations": sims,
                "raw_states": raw.get("states", []),
                "raw_err": raw.get("err"),
            }
        )
        if limit and len(rows) >= limit:
            break
    print(
        json.dumps(
            {
                "event": "calibration_loaded",
                "rows": len(rows),
                "label_counts": dict(Counter(str(row.get("gold_label")) for row in rows)),
                "skipped_test_overlap": skipped_overlap,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rows


def coverage(rows: List[Dict[str, Any]], budget: int) -> float:
    if not rows or budget <= 0:
        return 0.0
    return sum(1 for row in rows if row["simulations"][str(budget)]["normalized_status"] in DETERMINED) / len(rows)


def labeled_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("gold_label") in {"yes", "no"}]


def correct_coverage(rows: List[Dict[str, Any]], budget: int) -> float:
    part = labeled_rows(rows)
    if not part or budget <= 0:
        return 0.0
    return sum(1 for row in part if row["simulations"][str(budget)]["verieql_label"] == row["gold_label"]) / len(part)


def wrong_coverage(rows: List[Dict[str, Any]], budget: int) -> float:
    part = labeled_rows(rows)
    if not part or budget <= 0:
        return 0.0
    wrong = 0
    for row in part:
        label = row["simulations"][str(budget)]["verieql_label"]
        if label in {"yes", "no"} and label != row["gold_label"]:
            wrong += 1
    return wrong / len(part)


def undetermined_rate(rows: List[Dict[str, Any]], budget: int) -> float:
    part = labeled_rows(rows)
    if not part:
        return 1.0
    if budget <= 0:
        return 1.0
    determined = sum(1 for row in part if row["simulations"][str(budget)]["normalized_status"] in DETERMINED)
    return 1.0 - determined / len(part)


def avg_runtime(rows: List[Dict[str, Any]], budget: int) -> float:
    if not rows or budget <= 0:
        return 0.0
    return mean(float(row["simulations"][str(budget)]["runtime_s"]) for row in rows)


def available_positive_budgets(rows: List[Dict[str, Any]]) -> List[int]:
    if not rows:
        return []
    keys = set()
    for row in rows:
        keys.update(str(k) for k in (row.get("simulations") or {}).keys())
    out = []
    for key in keys:
        try:
            budget = int(key)
        except Exception:
            continue
        if budget > 0:
            out.append(budget)
    return sorted(set(out))


def clamp_budget(requested: int, budgets: List[int]) -> int:
    if not budgets:
        return 0
    candidates = [b for b in budgets if b <= requested]
    return max(candidates) if candidates else min(budgets)


def hybrid_utility(rows: List[Dict[str, Any]], budget: int, fallback_prior: float, latency_weight: float, scale_budget: int) -> float:
    part = labeled_rows(rows)
    if not part:
        return coverage(rows, budget)
    correct = correct_coverage(part, budget)
    unresolved = undetermined_rate(part, budget)
    latency_penalty = latency_weight * (avg_runtime(part, budget) / max(scale_budget, 1))
    return correct + fallback_prior * unresolved - latency_penalty


def unsupportedish_rate(rows: List[Dict[str, Any]], budget: int = 10) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["simulations"][str(budget)]["normalized_status"] in UNSUPPORTEDISH) / len(rows)


def mine_zero_flags(rows: List[Dict[str, Any]], min_support: int, precision_threshold: float, max_det_rate: float) -> Dict[str, Any]:
    by_flag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for flag in row.get("static_unsupported_flags", []):
            by_flag[str(flag)].append(row)
    flag_stats = {}
    learned = set()
    for flag, part in sorted(by_flag.items()):
        support = len(part)
        det10 = coverage(part, 10)
        uns10 = unsupportedish_rate(part, 10)
        source_backed = flag in ZERO_SOURCE_FLAGS
        selected = support >= min_support and ((uns10 >= precision_threshold and det10 <= max_det_rate) or (source_backed and uns10 >= 0.90 and det10 <= 0.03))
        if selected:
            learned.add(flag)
        flag_stats[flag] = {
            "support": support,
            "det_at_10": det10,
            "unsupportedish_at_10": uns10,
            "source_backed": source_backed,
            "selected_zero_flag": selected,
        }
    # Source-backed flags seen in test but rare in calibration still represent explicit VeriEQL limits.
    hard_zero = learned | {flag for flag in ZERO_SOURCE_FLAGS if flag in flag_stats and flag_stats[flag]["unsupportedish_at_10"] >= 0.90}
    return {"zero_flags": sorted(hard_zero), "flag_stats": flag_stats}


def choose_budget_for_group(rows: List[Dict[str, Any]], variant: Dict[str, Any]) -> int:
    if not rows:
        return 10
    positive_budgets = available_positive_budgets(rows)
    if not positive_budgets:
        return 0
    objective = str(variant.get("objective", "coverage"))
    if objective == "hybrid_utility":
        fallback_prior = float(variant.get("fallback_prior", 0.85))
        latency_weight = float(variant.get("latency_weight", 0.0))
        eps = float(variant.get("epsilon", 0.0))
        target_budget = clamp_budget(int(variant.get("target_budget", max(positive_budgets))), positive_budgets)
        candidate_budgets = [b for b in positive_budgets if b <= target_budget]
        if not labeled_rows(rows):
            candidate_budgets = candidate_budgets or [10]
            return min(candidate_budgets, key=lambda b: (b, avg_runtime(rows, b)))
        scored = [
            (
                budget,
                hybrid_utility(
                    rows,
                    budget,
                    fallback_prior=fallback_prior,
                    latency_weight=latency_weight,
                    scale_budget=max(positive_budgets),
                ),
            )
            for budget in candidate_budgets
        ]
        best_score = max(score for _, score in scored)
        # Among statistically tied budgets, keep the fastest one. This is the
        # core latency compression step rather than a test-set tweak.
        near_best = [budget for budget, score in scored if score >= best_score - eps]
        return min(near_best, key=lambda b: (avg_runtime(rows, b), b))

    eps = float(variant["epsilon"])
    target_budget = clamp_budget(int(variant["target_budget"]), positive_budgets)
    long_target_budget = clamp_budget(int(variant.get("long_target_budget", target_budget)), positive_budgets)
    long_gain_threshold = float(variant.get("long_gain_threshold", 1.0))
    metric = correct_coverage if objective == "correct_coverage" and labeled_rows(rows) else coverage
    target_cov = metric(rows, target_budget)
    long_cov = metric(rows, long_target_budget)
    if long_target_budget > target_budget and long_cov - target_cov >= long_gain_threshold:
        target_budget = long_target_budget
        target_cov = long_cov
    if target_cov <= 0 and unsupportedish_rate(rows, 10) >= 0.95:
        return 0
    for budget in positive_budgets:
        if budget > target_budget:
            continue
        if metric(rows, budget) >= target_cov - eps:
            return budget
    return target_budget


def summarize_group(key: str, part: List[Dict[str, Any]], selected: int, positive_budgets: List[int]) -> Dict[str, Any]:
    runtimes = [float(row["simulations"][str(selected)]["runtime_s"]) for row in part] if selected > 0 else [0.0 for _ in part]
    return {
        "key": key,
        "support": len(part),
        "labeled_support": len(labeled_rows(part)),
        "selected_budget": selected,
        "coverage_by_budget": {str(b): coverage(part, b) for b in positive_budgets},
        "correct_coverage_by_budget": {str(b): correct_coverage(part, b) for b in positive_budgets},
        "wrong_coverage_by_budget": {str(b): wrong_coverage(part, b) for b in positive_budgets},
        "unsupportedish_at_10": unsupportedish_rate(part, 10),
        "avg_runtime_at_selected": mean(runtimes),
        "p95_runtime_at_selected": percentile(runtimes, 95),
    }


def learn_policy(rows: List[Dict[str, Any]], variant_name: str, variant: Dict[str, Any], zero_flags: List[str]) -> Dict[str, Any]:
    positive_budgets = available_positive_budgets(rows)
    levels = ["fine", "medium", "coarse", "bucket"]
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {level: defaultdict(list) for level in levels}
    for row in rows:
        for level in levels:
            grouped[level][policy_key(row, level)].append(row)
    policy_levels = []
    min_support = variant["min_support"]
    for level in levels:
        groups = {}
        for key, part in grouped[level].items():
            if len(part) < int(min_support.get(level, 1)):
                continue
            selected = choose_budget_for_group(part, variant)
            groups[key] = summarize_group(key, part, selected, positive_budgets)
        policy_levels.append({"name": level, "min_support": int(min_support.get(level, 1)), "groups": groups})
    return {
        "name": variant_name,
        "description": variant["description"],
        "objective": variant.get("objective", "coverage"),
        "fallback_prior": variant.get("fallback_prior"),
        "latency_weight": variant.get("latency_weight"),
        "epsilon": variant.get("epsilon"),
        "budgets": [0, *positive_budgets],
        "sample_bound": 10,
        "strict_rule": "equivalent requires EQU at every bound; partial EQU followed by TMO is timeout",
        "zero_flags": zero_flags,
        "escalate_timeout_to_10_risk_flags": sorted(ESCALATE10_RISK_FLAGS),
        "fallback_order": levels,
        "default_budget": 10,
        "levels": policy_levels,
    }


def select_budget(descriptor: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    static_flags = set(str(x) for x in descriptor.get("static_unsupported_flags", []))
    zero_hits = sorted(static_flags & set(policy.get("zero_flags", [])))
    if zero_hits:
        return 0, {"policy_level": "zero_flag", "policy_key": "+".join(zero_hits), "route_reason": "explicit_unsupported_flag"}
    level_map = {level["name"]: level for level in policy["levels"]}
    for level_name in policy.get("fallback_order", []):
        level = level_map.get(level_name)
        if not level:
            continue
        key = policy_key(descriptor, level_name)
        group = level["groups"].get(key)
        if group:
            selected = int(group["selected_budget"])
            return selected, {
                "policy_level": level_name,
                "policy_key": key,
                "policy_support": group["support"],
                "policy_reference": group,
                "route_reason": f"profile_selected_{group['selected_budget']}s",
            }
    return int(policy.get("default_budget", 10)), {"policy_level": "default", "policy_key": "default", "route_reason": "no_matching_group"}


def load_fixed_budget_results(split: str, budgets: List[int], sample_bound: int) -> Dict[int, Dict[str, Dict[str, Any]]]:
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    base = PROJECT_ROOT / "outputs/verieql_budgeted"
    for budget in budgets:
        if budget <= 0:
            continue
        ready_path = base / f"{split}.fixed{budget}s.budget{budget}.verieql_ready.jsonl"
        raw_path = base / f"{split}.fixed{budget}s.budget{budget}.verieql_output.jsonl"
        if not ready_path.exists() or not raw_path.exists():
            continue
        ready_index = {int(row["index"]): row["id"] for row in read_jsonl(ready_path)}
        rows: Dict[str, Dict[str, Any]] = {}
        for raw in read_jsonl(raw_path):
            sid = ready_index.get(int(raw["index"])) if raw.get("index") is not None else raw.get("id")
            if not sid:
                continue
            sim = simulate_verieql_record(raw, budget, sample_bound)
            rows[str(sid)] = {
                "raw_states": raw.get("states", []),
                "raw_err": raw.get("err"),
                "normalized_status": sim["normalized_status"],
                "verieql_status": sim["normalized_status"],
                "verieql_label": sim["verieql_label"],
                "runtime_s": sim["runtime_s"],
                "verieql_runtime": sim["runtime_s"],
                "attempted_bounds": sim["attempted_bounds"],
                "observed_states": sim["observed_states"],
                "terminal_reason": sim["terminal_reason"],
            }
        out[budget] = rows
    return out


def load_standard(split: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(PROJECT_ROOT / "data/standard" / f"{split}.jsonl")}


def make_adaptive_stage(split: str, policy: Dict[str, Any], fixed_results: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    standard = load_standard(split)
    rows = []
    for sid, sample in standard.items():
        descriptor = descriptor_from_standard_row(sample)
        budget, route = select_budget(descriptor, policy)
        base = {
            "id": sid,
            "dataset": sample["dataset"],
            "gold_label": sample["label"],
            "bucket": descriptor["bucket"],
            "features": descriptor["features"],
            "selected_budget": budget,
            "static_unsupported_flags": descriptor.get("static_unsupported_flags", []),
            "risk_flags": descriptor.get("risk_flags", []),
            "strategy": policy["name"],
            "route": "zero_budget" if budget <= 0 else "run_verieql",
            **route,
        }
        if budget <= 0:
            rows.append(
                {
                    **base,
                    "raw_states": [],
                    "raw_err": route["route_reason"],
                    "normalized_status": "zero_budget_routed",
                    "verieql_status": "zero_budget_routed",
                    "verieql_label": None,
                    "runtime_s": 0.0,
                    "verieql_runtime": 0.0,
                    "fallback_triggered": True,
                }
            )
            continue
        raw = fixed_results.get(budget, {}).get(sid)
        if raw is None:
            rows.append(
                {
                    **base,
                    "raw_states": [],
                    "raw_err": f"missing_fixed{budget}s_result",
                    "normalized_status": "runtime_error",
                    "verieql_status": "runtime_error",
                    "verieql_label": None,
                    "runtime_s": float(budget),
                    "verieql_runtime": float(budget),
                    "fallback_triggered": True,
                }
            )
        else:
            risk_hits = sorted(set(descriptor.get("risk_flags", [])) & set(policy.get("escalate_timeout_to_10_risk_flags", [])))
            if 0 < budget < 10 and raw["verieql_status"] == "timeout" and risk_hits:
                followup = fixed_results.get(10, {}).get(sid)
                if followup is not None:
                    raw = {
                        **followup,
                        "runtime_s": float(raw.get("runtime_s", 0.0) or 0.0) + float(followup.get("runtime_s", 0.0) or 0.0),
                        "verieql_runtime": float(raw.get("verieql_runtime", 0.0) or 0.0) + float(followup.get("verieql_runtime", 0.0) or 0.0),
                        "initial_budget": budget,
                        "selected_budget": 10,
                        "budget_path": f"{budget}->10",
                        "escalation_reason": "timeout_with_refutation_risk:" + "+".join(risk_hits),
                    }
                    base["selected_budget"] = 10
                    base["initial_budget"] = budget
                    base["budget_path"] = f"{budget}->10"
                    base["route_reason"] = base.get("route_reason", "") + f"_escalate_on_timeout={'+'.join(risk_hits)}"
            status = raw["verieql_status"]
            rows.append({**base, **raw, "fallback_triggered": status not in DETERMINED})
    return rows


def load_logs(split: str, model_key: str) -> Dict[str, Dict[str, Any]]:
    candidates = [
        PROJECT_ROOT / "outputs/model_only_logs" / f"{split}.model_only.{model_key}.model_only.jsonl",
        PROJECT_ROOT / "outputs/fallback_logs" / f"{split}.adaptive_accuracy.{model_key}.fallback.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return {str(row["id"]): row for row in read_jsonl(path)}
    return {}


def fallback_label(log: Optional[Dict[str, Any]], strategy: str) -> Optional[str]:
    if not log:
        return None
    parsed = [x for x in log.get("parsed_answers", [])]
    valid = [x for x in parsed if x in {"yes", "no"}]
    if strategy == "direct":
        return parsed[0] if parsed and parsed[0] in {"yes", "no"} else None
    if strategy == "self_consistency":
        if valid and len(valid) == len(parsed) and len(set(valid)) == 1:
            return valid[0]
        return None
    if strategy == "majority":
        if not valid:
            return None
        counts = Counter(valid)
        if counts["yes"] == counts["no"]:
            return None
        return "yes" if counts["yes"] > counts["no"] else "no"
    raise ValueError(f"Unknown fallback strategy: {strategy}")


def verieql_label(status: str) -> Optional[str]:
    if status == "equivalent":
        return "yes"
    if status == "non_equivalent":
        return "no"
    return None


def evaluate_stage(stage_rows: List[Dict[str, Any]], logs: Dict[str, Dict[str, Any]], fallback_strategy: str, model_key: str) -> Dict[str, Any]:
    final_rows = []
    for stage in stage_rows:
        label = verieql_label(stage["verieql_status"])
        decision_source = "verified"
        if label is None:
            label = fallback_label(logs.get(stage["id"]), fallback_strategy)
            decision_source = "model_assisted" if label in {"yes", "no"} else "abstained"
        final_rows.append(
            {
                "id": stage["id"],
                "dataset": stage["dataset"],
                "gold_label": stage["gold_label"],
                "selected_budget": stage["selected_budget"],
                "verieql_status": stage["verieql_status"],
                "final_label": label if label in {"yes", "no"} else "uncertain",
                "decision_source": decision_source,
                "is_decided": label in {"yes", "no"},
                "is_correct": (label == stage["gold_label"]) if label in {"yes", "no"} else None,
                "verieql_runtime": float(stage.get("verieql_runtime", 0.0) or 0.0),
                "fallback_model_key": model_key,
                "fallback_strategy": fallback_strategy,
            }
        )
    n = len(final_rows)
    decided = [r for r in final_rows if r["is_decided"]]
    correct = [r for r in final_rows if r["is_correct"] is True]
    wrong = [r for r in final_rows if r["is_decided"] and r["is_correct"] is False]
    formal_times = [float(r["verieql_runtime"]) for r in final_rows]
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
        "avg_formal_runtime": mean(formal_times),
        "p95_formal_runtime": percentile(formal_times, 95),
        "budget_counts": dict(Counter(str(r["selected_budget"]) for r in final_rows)),
        "source_counts": dict(Counter(r["decision_source"] for r in final_rows)),
    }


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "policy",
        "model",
        "fallback",
        "overall_acc",
        "coverage",
        "accuracy_on_decided",
        "wrong_rate",
        "avg_formal_runtime",
        "p95_formal_runtime",
        "budget_counts",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|---" * len(fields) + "|\n")
        for row in rows:
            vals = []
            for field in fields:
                value = row.get(field)
                if isinstance(value, float):
                    vals.append(f"{value:.2%}" if field in {"overall_acc", "coverage", "accuracy_on_decided", "wrong_rate"} else f"{value:.2f}")
                elif isinstance(value, dict):
                    vals.append("`" + json.dumps(value, ensure_ascii=False, sort_keys=True) + "`")
                else:
                    vals.append(str(value))
            f.write("| " + " | ".join(vals) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--limit-calibration", type=int, default=None)
    ap.add_argument("--budgets", default=None, help="Comma-separated budgets to use, e.g. 0,1,2,3,5,10,30")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sample_bound = int(cfg["verieql"]["sample_bound"])
    if args.budgets:
        budgets = sorted(set(int(x.strip()) for x in args.budgets.split(",") if x.strip()))
    else:
        budgets = [0, *POSITIVE_TEST_BUDGETS]
    positive_budgets = [b for b in budgets if b > 0]
    if not positive_budgets:
        raise ValueError("At least one positive VeriEQL budget is required.")
    cal_rows = load_calibration_rows(cfg, budgets, args.limit_calibration)
    zero_info = mine_zero_flags(cal_rows, min_support=20, precision_threshold=0.98, max_det_rate=0.01)

    out_dir = PROJECT_ROOT / "outputs/adaptive_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "zero_flag_mining.json", zero_info)

    policies = {
        name: learn_policy(cal_rows, name, variant, zero_info["zero_flags"])
        for name, variant in VARIANTS.items()
    }
    for name, policy in policies.items():
        write_json(out_dir / f"{name}.policy.json", policy)

    models = [str(m["key"]) for m in cfg["fallback"].get("models", []) if m.get("enabled", True)]
    fallback_strategies = ["direct", "self_consistency", "majority"]
    all_metrics = []
    outputs: Dict[str, Any] = {"policies": {}, "stages": {}, "metrics": {}}
    for split in cfg.get("experiments", {}).get("test_splits", ["test_leetcode", "test_calcite_spider"]):
        fixed_results = load_fixed_budget_results(split, positive_budgets, sample_bound)
        for policy_name, policy in policies.items():
            stage_rows = make_adaptive_stage(split, policy, fixed_results)
            stage_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{policy_name}.verieql_stage.jsonl"
            write_jsonl(stage_path, stage_rows)
            outputs["stages"][f"{split}:{policy_name}"] = str(stage_path)
            for model_key in models:
                logs = load_logs(split, model_key)
                for fallback_strategy in fallback_strategies:
                    metrics = evaluate_stage(stage_rows, logs, fallback_strategy, model_key)
                    record = {
                        "dataset": split,
                        "policy": policy_name,
                        "model": model_key,
                        "fallback": fallback_strategy,
                        **metrics,
                    }
                    all_metrics.append(record)
    write_json(out_dir / "adaptive_offline_metrics.json", all_metrics)
    csv_path = out_dir / "adaptive_offline_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    write_markdown(PROJECT_ROOT / "outputs/tables/adaptive_offline_search.md", all_metrics)
    outputs["metrics"] = {"json": str(out_dir / "adaptive_offline_metrics.json"), "csv": str(csv_path)}
    update_manifest(PROJECT_ROOT / "outputs", "adaptive_offline_search", outputs)
    print(json.dumps({"metrics": outputs["metrics"], "rows": len(all_metrics), "zero_flags": zero_info["zero_flags"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
