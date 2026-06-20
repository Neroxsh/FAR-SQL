#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


REFUTED_STATES = {"NEQ", "SAT", "REFUTED"}
CHECKED_STATES = {"EQU", "UNSAT", "CHECKED", "VERIFIED"}
TIMEOUT_STATES = {"TMO", "TIMEOUT"}
UNSUPPORTED_STATES = {"NSE", "NIE"}
CONVERSION_STATES = {"SYN"}
RUNTIME_ERROR_STATES = {"OOM", "OTE"}
UNKNOWN_STATES = {"UNK"}
DETERMINED = {"equivalent", "non_equivalent"}


def flatten_times(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isnan(v):
            out.append(v)
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_times(item))
    return out


def elapsed_for_state(value: Any, state: str, timeout_sec: float) -> float:
    """Return elapsed wall-like verifier effort for one bound attempt.

    VeriEQL records timeout bounds as either None or [T, T]. The latter is not
    two sequential T-second phases; it is one killed bound, so it costs T.
    """
    state = str(state).upper()
    if state in TIMEOUT_STATES:
        return float(timeout_sec)
    vals = flatten_times(value)
    if not vals:
        return 0.0
    return float(sum(vals))


def terminal_status_from_state(row: Dict[str, Any], state: str) -> Tuple[Optional[str], Optional[str]]:
    state = str(state).upper()
    err = str(row.get("err", "") or "")
    low = err.lower()
    if state in REFUTED_STATES:
        return "non_equivalent", "no"
    if state in TIMEOUT_STATES:
        return "timeout", None
    if state in UNSUPPORTED_STATES:
        return "unsupported_runtime", None
    if state in CONVERSION_STATES:
        return "conversion_error", None
    if state in UNKNOWN_STATES:
        return "unknown", None
    if state in RUNTIME_ERROR_STATES:
        return "runtime_error", None
    if state in CHECKED_STATES:
        return None, None
    if "not equivalent" in low or row.get("counterexample"):
        return "non_equivalent", "no"
    if "time out" in low or "timeout" in low:
        return "timeout", None
    if "not supported feature" in low or "not implemented" in low or "not supported" in low:
        return "unsupported_runtime", None
    if "syntaxerror" in low or "parsersyntaxerror" in low or "unknowncolumn" in low or "unknowndatabase" in low:
        return "conversion_error", None
    if "unknown" in low or "undecidable" in low:
        return "unknown", None
    if "exception" in low or "traceback" in low:
        return "runtime_error", None
    return None, None


def simulate_verieql_record(row: Dict[str, Any], timeout_sec: int | float, sample_bound: int = 10) -> Dict[str, Any]:
    """Simulate strict `cli_within_bound -s sample_bound -t timeout_sec`.

    Strict equivalence requires all `sample_bound` attempts to finish as EQU.
    Partial sequences such as [EQU, EQU, TMO] are timeout, not equivalent.
    A refutation can stop early as non-equivalent.
    """
    timeout = float(timeout_sec)
    states = [str(x).upper() for x in row.get("states", []) if x is not None]
    times = row.get("times") if isinstance(row.get("times"), list) else []
    elapsed = 0.0
    observed_states: List[str] = []

    for idx in range(sample_bound):
        if idx >= len(states):
            elapsed += timeout
            observed_states.append("TMO")
            return {
                "normalized_status": "timeout",
                "verieql_label": None,
                "runtime_s": elapsed,
                "attempted_bounds": len(observed_states),
                "observed_states": observed_states,
                "terminal_reason": "missing_bound_treated_as_timeout",
            }

        state = states[idx]
        bound_time = elapsed_for_state(times[idx] if idx < len(times) else None, state, timeout)
        if state not in TIMEOUT_STATES and bound_time > timeout:
            elapsed += timeout
            observed_states.append("TMO")
            return {
                "normalized_status": "timeout",
                "verieql_label": None,
                "runtime_s": elapsed,
                "attempted_bounds": len(observed_states),
                "observed_states": observed_states,
                "terminal_reason": "simulated_timeout_before_recorded_state",
            }

        elapsed += bound_time
        observed_states.append(state)
        status, label = terminal_status_from_state(row, state)
        if status is not None:
            return {
                "normalized_status": status,
                "verieql_label": label,
                "runtime_s": elapsed,
                "attempted_bounds": len(observed_states),
                "observed_states": observed_states,
                "terminal_reason": f"terminal_state_{state}",
            }

        if state in CHECKED_STATES:
            if idx == sample_bound - 1:
                return {
                    "normalized_status": "equivalent",
                    "verieql_label": "yes",
                    "runtime_s": elapsed,
                    "attempted_bounds": len(observed_states),
                    "observed_states": observed_states,
                    "terminal_reason": "all_bounds_checked",
                }
            continue

        return {
            "normalized_status": "runtime_error",
            "verieql_label": None,
            "runtime_s": elapsed,
            "attempted_bounds": len(observed_states),
            "observed_states": observed_states,
            "terminal_reason": f"unrecognized_state_{state}",
        }

    return {
        "normalized_status": "unknown",
        "verieql_label": None,
        "runtime_s": elapsed,
        "attempted_bounds": len(observed_states),
        "observed_states": observed_states,
        "terminal_reason": "fallthrough",
    }
