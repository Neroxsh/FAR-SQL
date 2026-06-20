#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import PROJECT_ROOT, load_config, read_jsonl, update_manifest, write_jsonl


TEST_SPLITS = ["test_leetcode", "test_calcite_spider"]
DETERMINED = {"equivalent", "non_equivalent"}
SYSTEM_PROMPT = "You are a helpful SQL expert."
INSTRUCTION_PROMPT = """You are an SQL expert tasked with determining whether two SQL queries are equivalent (i.e., produce identical results when executed). Please carefully analyze the database schema and the two SQL queries. Reason step by step to determine whether the two SQL queries are equivalent. Let’s think through this step by step.

Output Requirements:
1. Provide a detailed explanation of your reasoning process within the <thinking></thinking> tags.
2. Output only "yes" or "no" within the <answer> </answer> tags."""


def load_standard(split: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(PROJECT_ROOT / "data/standard" / f"{split}.jsonl")}


def format_schema(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    formatted_input = json.dumps(
        {
            "sql1": sample["sql1"],
            "sql2": sample["sql2"],
            "schema": sample.get("schema", {}),
        },
        ensure_ascii=False,
        indent=2,
    )
    return INSTRUCTION_PROMPT + "\n\n" + formatted_input


def build_training_with_verifier_observation_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    observed = [str(x).upper() for x in (stage.get("observed_states") or stage.get("raw_states") or [])]
    compact_observation = {
        "verifier_status": stage.get("verieql_status"),
        "selected_budget_seconds": stage.get("selected_budget"),
        "runtime_seconds": stage.get("verieql_runtime"),
        "bucket": stage.get("bucket"),
        "risk_flags": stage.get("risk_flags", []),
        "verifier_observation": compact_verieql_hint(stage),
        "observed_states": observed[-6:],
    }
    formatted_input = json.dumps(
        {
            "sql1": sample["sql1"],
            "sql2": sample["sql2"],
            "schema": sample.get("schema", {}),
            "auxiliary_verifier_observation": compact_observation,
        },
        ensure_ascii=False,
        indent=2,
    )
    guidance = (
        "\n\nAdditional note: The auxiliary_verifier_observation field is weak evidence only. "
        "Do not treat verifier failure itself as proof of non-equivalence. "
        "Judge mainly from SQL semantics."
    )
    return INSTRUCTION_PROMPT + guidance + "\n\n" + formatted_input


def build_contextual_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    features = stage.get("features") or {}
    payload = {
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "schema": sample.get("schema", {}),
        "verieql_status": stage.get("verieql_status"),
        "bucket": stage.get("bucket"),
        "sql_features": {
            "join_count": features.get("join_count"),
            "nesting_depth": features.get("nesting_depth"),
            "has_aggregation": features.get("has_aggregation"),
            "has_group_by": features.get("has_group_by"),
            "has_distinct": features.get("has_distinct"),
            "has_null_predicate": features.get("has_null_predicate"),
            "has_set_op": features.get("has_set_op"),
        },
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_verieql_aware_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent. VeriEQL was attempted first but did not return a final deterministic equivalence decision for this sample.",
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "schema": sample.get("schema", {}),
        "verieql_result": {
            "status": stage.get("verieql_status"),
            "selected_budget_seconds": stage.get("selected_budget"),
            "runtime_seconds": stage.get("verieql_runtime"),
            "states": stage.get("observed_states") or stage.get("raw_states", []),
            "error": stage.get("raw_err"),
            "route_reason": stage.get("route_reason"),
            "static_unsupported_flags": stage.get("static_unsupported_flags", []),
            "risk_flags": stage.get("risk_flags", []),
        },
        "important_note": "Do not assume the VeriEQL failure means equivalent or non-equivalent. Timeout/unsupported/error only means the formal verifier did not decide. Make your own SQL semantic judgment.",
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_verieql_packaged_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    observed = list(stage.get("observed_states") or stage.get("raw_states") or [])
    verifier_summary = {
        "status": stage.get("verieql_status"),
        "selected_budget_seconds": stage.get("selected_budget"),
        "runtime_seconds": stage.get("verieql_runtime"),
        "observed_states": observed,
        "error": stage.get("raw_err"),
        "bucket": stage.get("bucket"),
        "risk_flags": stage.get("risk_flags", []),
        "static_unsupported_flags": stage.get("static_unsupported_flags", []),
    }
    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent.",
        "schema": sample.get("schema", {}),
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "verieql_attempt": verifier_summary,
        "how_to_use_verieql_signal": [
            "Use the SQL semantics as the main evidence.",
            "The VeriEQL trace is auxiliary evidence and may still be useful even when VeriEQL did not finish with a final decision.",
            "If the trace shows many EQU-like states before timeout, treat that as weak support for equivalence, but not as a proof by itself.",
            "If the failure looks like unsupported syntax or runtime failure, do not treat that failure itself as evidence of non-equivalence.",
            "Answer no only when you can identify a concrete semantic mismatch such as duplicate behavior, NULL behavior, grouping level mismatch, filter scope mismatch, join multiplicity difference, set-operation difference, ordering/limit difference, or correlated-subquery difference.",
        ],
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def summarize_verieql_signal(stage: Dict[str, Any]) -> Dict[str, Any]:
    observed = [str(x).upper() for x in (stage.get("observed_states") or stage.get("raw_states") or [])]
    status = str(stage.get("verieql_status") or "unknown")
    equ_prefix = 0
    saw_neq = False
    saw_timeout = False
    saw_runtime_like = False
    saw_unsupported_like = False
    for token in observed:
        if token == "NEQ":
            saw_neq = True
        if token == "TMO":
            saw_timeout = True
        if token in {"OTE"}:
            saw_runtime_like = True
        if token in {"NIE"}:
            saw_unsupported_like = True
        if token == "EQU":
            equ_prefix += 1
        else:
            break
    only_equ_then_timeout = bool(observed) and observed[-1] == "TMO" and all(tok == "EQU" for tok in observed[:-1])
    return {
        "status": status,
        "budget_seconds": stage.get("selected_budget"),
        "runtime_seconds": stage.get("verieql_runtime"),
        "bucket": stage.get("bucket"),
        "risk_flags": stage.get("risk_flags", []),
        "static_unsupported_flags": stage.get("static_unsupported_flags", []),
        "equ_prefix_count": equ_prefix,
        "trace_length": len(observed),
        "only_equ_then_timeout": only_equ_then_timeout,
        "saw_neq_state": saw_neq,
        "saw_timeout_state": saw_timeout,
        "saw_runtime_like_state": saw_runtime_like,
        "saw_unsupported_like_state": saw_unsupported_like,
        "error": stage.get("raw_err"),
    }


def build_verieql_summary_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    base_payload = {
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "schema": sample.get("schema", {}),
    }
    signal = summarize_verieql_signal(stage)
    evidence_lines = [
        "VeriEQL was attempted first but did not give a final yes/no answer.",
        f"Final verifier status: {signal['status']}.",
        f"Assigned verifier budget: {signal['budget_seconds']} seconds.",
        f"Observed verifier runtime: {signal['runtime_seconds']} seconds.",
        f"SQL bucket: {signal['bucket']}.",
    ]
    if signal["only_equ_then_timeout"]:
        evidence_lines.append(
            "Before timeout, the verifier trace only showed EQU states. This is weak support for equivalence, but not a proof."
        )
    elif signal["equ_prefix_count"] > 0:
        evidence_lines.append(
            f"The verifier trace began with {signal['equ_prefix_count']} EQU state(s), but the run still ended without a final decision."
        )
    if signal["saw_neq_state"]:
        evidence_lines.append(
            "The verifier trace contained an NEQ state. Treat this as important warning evidence and check carefully for a concrete semantic mismatch."
        )
    if signal["saw_runtime_like_state"]:
        evidence_lines.append(
            "The verifier also showed runtime-failure-like behavior. Do not treat the failure itself as proof of non-equivalence."
        )
    if signal["saw_unsupported_like_state"] or signal["static_unsupported_flags"]:
        evidence_lines.append(
            "The verifier appears to have limited support for this sample. Unsupported behavior is not evidence of non-equivalence."
        )
    if signal["risk_flags"]:
        evidence_lines.append("Potential SQL risk factors: " + ", ".join(str(x) for x in signal["risk_flags"]) + ".")
    if signal["static_unsupported_flags"]:
        evidence_lines.append(
            "Potential unsupported syntax markers: " + ", ".join(str(x) for x in signal["static_unsupported_flags"]) + "."
        )
    if signal["error"]:
        evidence_lines.append(f"Verifier message: {signal['error']}")
    evidence_lines.extend(
        [
            "Use SQL semantics as the main evidence.",
            "Answer no only when you can identify a concrete semantic mismatch such as duplicate behavior, NULL behavior, grouping level mismatch, filter scope mismatch, join multiplicity difference, set-operation difference, ordering/limit difference, or correlated-subquery difference.",
        ]
    )
    return (
        INSTRUCTION_PROMPT
        + "\n\n"
        + json.dumps(base_payload, ensure_ascii=False, indent=2)
        + "\n\nAuxiliary VeriEQL evidence:\n- "
        + "\n- ".join(evidence_lines)
    )


def compact_verieql_hint(stage: Dict[str, Any]) -> Dict[str, str]:
    observed = [str(x).upper() for x in (stage.get("observed_states") or stage.get("raw_states") or [])]
    status = str(stage.get("verieql_status") or "unknown")
    saw_neq = "NEQ" in observed
    only_equ_then_timeout = bool(observed) and observed[-1] == "TMO" and all(tok == "EQU" for tok in observed[:-1])
    if status == "timeout" and only_equ_then_timeout:
        return {
            "formal_hint": "weak_equivalence_hint",
            "explanation": "VeriEQL timed out, but before timeout it only emitted EQU states. This is weak support for equivalence, not a proof.",
        }
    if status == "timeout" and saw_neq:
        return {
            "formal_hint": "weak_nonequivalence_warning",
            "explanation": "VeriEQL did not finish, but its trace contained an NEQ state. Check carefully for a concrete semantic mismatch.",
        }
    if status in {"runtime_error", "conversion_error", "unsupported_runtime", "unknown"}:
        return {
            "formal_hint": "tool_failure_no_direction",
            "explanation": "VeriEQL failed because of tool/runtime/support limitations. This gives no direct yes/no direction.",
        }
    return {
        "formal_hint": "neutral",
        "explanation": "VeriEQL did not provide a decisive directional signal.",
    }


def build_verieql_hint_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    base_payload = {
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "schema": sample.get("schema", {}),
    }
    hint = compact_verieql_hint(stage)
    hint_block = [
        "Auxiliary formal hint from VeriEQL:",
        f"- Hint type: {hint['formal_hint']}",
        f"- Explanation: {hint['explanation']}",
        "- First decide from SQL semantics itself.",
        "- Use the formal hint only as weak auxiliary evidence.",
        "- Answer no only if you can identify a concrete semantic mismatch.",
    ]
    return (
        INSTRUCTION_PROMPT
        + "\n\n"
        + json.dumps(base_payload, ensure_ascii=False, indent=2)
        + "\n\n"
        + "\n".join(hint_block)
    )


def trace_summary(stage: Dict[str, Any]) -> Dict[str, Any]:
    observed = list(stage.get("observed_states") or stage.get("raw_states") or [])
    prefix_equ = 0
    for token in observed:
        if str(token).upper() in {"EQU", "UNSAT", "CHECKED", "VERIFIED"}:
            prefix_equ += 1
        else:
            break
    return {
        "status": stage.get("verieql_status"),
        "selected_budget_seconds": stage.get("selected_budget"),
        "runtime_seconds": stage.get("verieql_runtime"),
        "observed_states": observed,
        "prefix_equ_count": prefix_equ,
        "error": stage.get("raw_err"),
        "route_reason": stage.get("route_reason"),
        "risk_flags": stage.get("risk_flags", []),
        "static_unsupported_flags": stage.get("static_unsupported_flags", []),
    }


def build_trace_guided_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    trace = trace_summary(stage)
    guidance: List[str] = [
        "Use the SQL semantics as the primary evidence.",
        "Do not treat a verifier failure as a final proof by itself.",
    ]
    status = str(stage.get("verieql_status"))
    if status == "timeout":
        guidance.append(
            "The formal verifier timed out after partial checking. If the observed trace is mostly or entirely equivalence-preserving states before timeout, treat that as weak positive evidence toward equivalence, but only if the SQL semantics also look aligned."
        )
        guidance.append(
            "Output no only if you can identify a concrete semantic mismatch such as different duplicate behavior, join multiplicity, filter scope, NULL behavior, aggregation grain, set semantics, ordering/limit effects, or correlated-subquery logic."
        )
    elif status in {"conversion_error", "unsupported_runtime", "runtime_error"}:
        guidance.append(
            "The verifier failure is about support/runtime rather than a proved semantic difference. Do not overinterpret the failure message."
        )
        guidance.append(
            "Output no only if you can point to a real semantic difference between the two SQL queries."
        )
    else:
        guidance.append("Make an independent semantic judgment from the schema and SQL.")

    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent.",
        "schema": sample.get("schema", {}),
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "verifier_trace_summary": trace,
        "decision_guidance": guidance,
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_witness_guided_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent.",
        "schema": sample.get("schema", {}),
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "verifier_status": stage.get("verieql_status"),
        "rule": [
            "Answer no only if you can explain a concrete semantic difference or a plausible counterexample database situation.",
            "Examples of valid no-reasons include duplicate multiplicity differences, DISTINCT effects, GROUP BY level mismatch, aggregate scope mismatch, NULL semantics mismatch, join-key mismatch, filter placement mismatch, set-operation mismatch, limit/order differences, or correlated-subquery differences.",
            "If you cannot identify a concrete semantic mismatch, prefer yes.",
        ],
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_case_runtime_semantic_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    features = stage.get("features") or {}
    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent.",
        "schema": sample.get("schema", {}),
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "verifier_observation": {
            "status": stage.get("verieql_status"),
            "observed_states": stage.get("observed_states") or stage.get("raw_states", []),
            "selected_budget_seconds": stage.get("selected_budget"),
            "runtime_seconds": stage.get("verieql_runtime"),
            "error": stage.get("raw_err"),
        },
        "sql_structure": {
            "has_case_when": features.get("has_case_when"),
            "has_aggregation": features.get("has_aggregation"),
            "has_group_by": features.get("has_group_by"),
            "has_set_op": features.get("has_set_op"),
            "has_null_predicate": features.get("has_null_predicate"),
            "risk_flags": stage.get("risk_flags", []),
        },
        "decision_guidance": [
            "VeriEQL runtime_error means the formal tool failed internally; it is not evidence that the two SQL queries are non-equivalent.",
            "CASE/conditional aggregation queries are often written in different but equivalent forms. Compare the branch conditions, grouping keys, aggregate scope, NULL handling, duplicate behavior, and join multiplicity directly.",
            "Answer no only if you can identify a concrete semantic mismatch. Do not answer no merely because the verifier failed.",
            "If both queries implement the same CASE conditions, same grouping grain, and same aggregate values under all database instances, answer yes.",
        ],
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_counterexample_review_prompt(sample: Dict[str, Any], stage: Dict[str, Any]) -> str:
    features = stage.get("features") or {}
    observed = [str(x).upper() for x in (stage.get("observed_states") or stage.get("raw_states") or [])]
    payload = {
        "task": "Determine whether sql1 and sql2 are semantically equivalent.",
        "schema": sample.get("schema", {}),
        "sql1": sample["sql1"],
        "sql2": sample["sql2"],
        "verifier_observation": {
            "status": stage.get("verieql_status"),
            "observed_states": observed[-8:],
            "selected_budget_seconds": stage.get("selected_budget"),
            "runtime_seconds": stage.get("verieql_runtime"),
            "error": stage.get("raw_err"),
        },
        "sql_structure": {
            "bucket": stage.get("bucket"),
            "has_case_when": features.get("has_case_when"),
            "has_aggregation": features.get("has_aggregation"),
            "has_group_by": features.get("has_group_by"),
            "has_distinct": features.get("has_distinct"),
            "has_null_predicate": features.get("has_null_predicate"),
            "has_set_op": features.get("has_set_op"),
            "risk_flags": stage.get("risk_flags", []),
            "static_unsupported_flags": stage.get("static_unsupported_flags", []),
        },
        "review_protocol": [
            "First compare the two queries at the level of projected attributes, filters, joins, grouping keys, aggregate scope, duplicate semantics, NULL behavior, set operations, and limit/order behavior.",
            "If CASE WHEN or conditional aggregation appears, compare each branch condition and each aggregate expression directly. Different syntactic forms may still be equivalent.",
            "Treat VeriEQL timeout, unsupported syntax, or runtime_error as tool incompleteness only. They are not evidence for either yes or no.",
            "To answer no, you must identify a concrete semantic mismatch or a plausible database instance where the outputs differ.",
            "If no concrete mismatch can be identified after the checks above, answer yes.",
        ],
    }
    return INSTRUCTION_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--prompt-variant", choices=["training_compatible", "training_with_verifier_observation", "contextual", "verieql_aware", "verieql_packaged", "verieql_summary", "verieql_hint", "trace_guided", "witness_guided", "case_runtime_semantic", "counterexample_review"], default=None)
    ap.add_argument("--prompt-dir", default="outputs/fallback_prompts")
    ap.add_argument("--output-suffix", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.strategy is None:
        args.strategy = str(cfg.get("experiments", {}).get("autobudget_strategy_name", "autobudget"))
    prompt_variant = args.prompt_variant or str(cfg.get("fallback", {}).get("prompt_variant", "training_compatible"))
    outputs = {}
    for split in TEST_SPLITS:
        standard = load_standard(split)
        stage_path = PROJECT_ROOT / "outputs/decisions" / f"{split}.{args.strategy}.verieql_stage.jsonl"
        prompts = []
        for stage in read_jsonl(stage_path):
            if stage["verieql_status"] in DETERMINED:
                continue
            sample = standard[stage["id"]]
            prompts.append(
                {
                    "id": stage["id"],
                    "dataset": split,
                    "gold_label": stage["gold_label"],
                    "verieql_status": stage["verieql_status"],
                    "bucket": stage.get("bucket"),
                    "risk_flags": stage.get("risk_flags", []),
                    "static_unsupported_flags": stage.get("static_unsupported_flags", []),
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt_variant": prompt_variant,
                    "prompt": (
                        build_contextual_prompt(sample, stage)
                        if prompt_variant == "contextual"
                        else build_training_with_verifier_observation_prompt(sample, stage)
                        if prompt_variant == "training_with_verifier_observation"
                        else build_verieql_aware_prompt(sample, stage)
                        if prompt_variant == "verieql_aware"
                        else build_verieql_packaged_prompt(sample, stage)
                        if prompt_variant == "verieql_packaged"
                        else build_verieql_summary_prompt(sample, stage)
                        if prompt_variant == "verieql_summary"
                        else build_verieql_hint_prompt(sample, stage)
                        if prompt_variant == "verieql_hint"
                        else build_trace_guided_prompt(sample, stage)
                        if prompt_variant == "trace_guided"
                        else build_witness_guided_prompt(sample, stage)
                        if prompt_variant == "witness_guided"
                        else build_case_runtime_semantic_prompt(sample, stage)
                        if prompt_variant == "case_runtime_semantic"
                        else build_counterexample_review_prompt(sample, stage)
                        if prompt_variant == "counterexample_review"
                        else build_prompt(sample, stage)
                    ),
                }
            )
        filename = f"{split}.{args.strategy}.prompts.jsonl" if not args.output_suffix else f"{split}.{args.strategy}.{args.output_suffix}.prompts.jsonl"
        out_path = PROJECT_ROOT / args.prompt_dir / filename
        outputs[split] = {"path": str(out_path), "rows": write_jsonl(out_path, prompts)}
    update_manifest(PROJECT_ROOT / "outputs", "build_fallback_prompts", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
