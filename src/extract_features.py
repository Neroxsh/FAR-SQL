#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import PROJECT_ROOT, load_config, read_jsonl, resolve_path, standard_paths, update_manifest, write_jsonl


AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG|ARRAY_AGG)\s*\(", re.I)
FUNC_RE = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*\(", re.I)
SET_OP_RE = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.I)
JOIN_RE = re.compile(r"\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s+JOIN\b", re.I)
TABLE_AFTER_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_$]*|`[^`]+`)", re.I)
PRED_RE = re.compile(r"\b(WHERE|HAVING|AND|OR|=|<>|!=|>=|<=|>|<|LIKE|BETWEEN|IN|EXISTS)\b", re.I)


def strip_strings(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def nesting_depth(sql: str) -> int:
    s = strip_strings(sql)
    depth = max_depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
            tail = s[i + 1 : i + 12].lstrip().upper()
            if tail.startswith("SELECT"):
                max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def select_clause(sql: str) -> str:
    s = strip_strings(sql)
    m = re.search(r"\bSELECT\b", s, re.I)
    if not m:
        return ""
    depth = 0
    for i in range(m.end(), len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and re.match(r"\sFROM\b", s[i:], re.I):
            return s[m.end() : i]
    return s[m.end() :]


def has_select_subquery(sql: str) -> bool:
    return bool(re.search(r"\(\s*SELECT\b", select_clause(sql), re.I))


def query_has_no_from(sql: str) -> bool:
    s = strip_strings(sql)
    return bool(re.search(r"\bSELECT\b", s, re.I)) and not bool(re.search(r"\bFROM\b", s, re.I))


def has_nested_case(sql: str) -> bool:
    tokens = re.findall(r"\bCASE\b|\bEND\b", strip_strings(sql), flags=re.I)
    depth = 0
    for token in tokens:
        t = token.upper()
        if t == "CASE":
            if depth > 0:
                return True
            depth += 1
        elif t == "END":
            depth = max(0, depth - 1)
    return False


def constraint_count(schema: Any, constraint: Any) -> int:
    if isinstance(constraint, list):
        return len(constraint)
    if isinstance(schema, str):
        return len(re.findall(r"\b(PRIMARY\s+KEY|FOREIGN\s+KEY|REFERENCES|UNIQUE)\b", schema, re.I))
    if isinstance(schema, dict):
        text = json.dumps(schema, ensure_ascii=False)
        return len(re.findall(r"\b(primary_keys|foreign_keys|PRIMARY|FOREIGN|REFERENCES)\b", text, re.I))
    return 0


def features_one(sql: str) -> Dict[str, Any]:
    s = strip_strings(sql)
    up = s.upper()
    tables = set(x.strip("`").upper() for x in TABLE_AFTER_RE.findall(s))
    return {
        "sql_length": len(re.findall(r"\S+", s)),
        "table_count": len(tables),
        "join_count": len(JOIN_RE.findall(s)),
        "predicate_count": len(PRED_RE.findall(s)),
        "nesting_depth": nesting_depth(s),
        "has_aggregation": bool(AGG_RE.search(s)),
        "has_group_by": bool(re.search(r"\bGROUP\s+BY\b", up)),
        "has_having": bool(re.search(r"\bHAVING\b", up)),
        "has_distinct": bool(re.search(r"\bDISTINCT\b", up)),
        "has_set_op": bool(SET_OP_RE.search(s)),
        "has_order_by": bool(re.search(r"\bORDER\s+BY\b", up)),
        "has_null_predicate": bool(re.search(r"\b(IS\s+NULL|IS\s+NOT\s+NULL|NOT\s+IN)\b", up)),
        "has_case_when": bool(re.search(r"\bCASE\b", up)),
    }


def merge_features(a: Dict[str, Any], b: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    bool_keys = [
        "has_aggregation",
        "has_group_by",
        "has_having",
        "has_distinct",
        "has_set_op",
        "has_order_by",
        "has_null_predicate",
        "has_case_when",
    ]
    out = {
        "sql_length": max(a["sql_length"], b["sql_length"]),
        "table_count": max(a["table_count"], b["table_count"]),
        "join_count": max(a["join_count"], b["join_count"]),
        "predicate_count": max(a["predicate_count"], b["predicate_count"]),
        "nesting_depth": max(a["nesting_depth"], b["nesting_depth"]),
        "constraint_count": constraint_count(row.get("schema"), row.get("constraint")),
    }
    for k in bool_keys:
        out[k] = bool(a[k] or b[k])
    return out


def assign_bucket(f: Dict[str, Any]) -> str:
    complex_flags = 0
    if f["has_aggregation"] or f["has_group_by"] or f["has_having"]:
        complex_flags += 1
    if f["nesting_depth"] > 0 or f["has_set_op"]:
        complex_flags += 1
    if f["has_distinct"] or f["has_null_predicate"] or f["has_case_when"]:
        complex_flags += 1
    if f["join_count"] >= 3 or f["table_count"] >= 4:
        complex_flags += 1
    if complex_flags >= 2:
        return "Complex-Mixed"
    if f["nesting_depth"] > 0 or f["has_set_op"]:
        return "Nested/Set"
    if f["has_aggregation"] or f["has_group_by"] or f["has_having"]:
        return "Aggregation"
    return "Simple-SPJ"


def flags_for_pair(sql1: str, sql2: str) -> Tuple[List[str], List[str]]:
    text = f"{sql1}\n{sql2}"
    up = strip_strings(text).upper()
    static = []
    risk = []
    if re.search(r"\bOVER\s*\(", up):
        static.append("window_over")
    if re.search(r"\bEXISTS\b", up):
        static.append("exists")
    if has_select_subquery(sql1) or has_select_subquery(sql2):
        static.append("select_subquery")
    if re.search(r"\b(STDDEV_POP|VAR_POP|STDDEV_SAMP|VAR_SAMP)\s*\(", up):
        static.append("unsupported_statistical_agg")
    if re.search(r"\b(GROUPING\s*\(|GROUPING\s+SETS|ROLLUP\s*\(|CUBE\s*\(|FILTER\s*\()\b", up):
        static.append("advanced_grouping_or_filter")
    if re.search(r"\bROW\s*\(", up):
        static.append("row_constructor")
    if re.search(r"\bVALUES\s*\(", up):
        static.append("values_clause")
    if has_nested_case(sql1) or has_nested_case(sql2):
        static.append("nested_case")
    if query_has_no_from(sql1) or query_has_no_from(sql2):
        static.append("select_without_from")
    if re.search(r"\bTIMESTAMPDIFF\s*\(\s*(MONTH|YEAR|HOUR|MINUTE|SECOND)\b", up):
        static.append("non_day_timestampdiff")
    if re.search(r"\bINTERVAL\s+[^,\)]*\b(MONTH|YEAR|HOUR|MINUTE|SECOND)\b", up):
        static.append("non_day_interval")

    risk_patterns = {
        "exists": r"\bEXISTS\b",
        "correlated_or_subquery": r"\(\s*SELECT\b",
        "union_all": r"\bUNION\s+ALL\b",
        "limit_offset": r"\b(LIMIT|OFFSET)\b",
        "grouping_sets": r"\bGROUPING\s+SETS\b|\bGROUPING\s*\(",
        "rollup": r"\bROLLUP\s*\(",
        "cube": r"\bCUBE\s*\(",
        "filter_clause": r"\bFILTER\s*\(",
        "cast": r"\bCAST\s*\(",
        "date_time_function": r"\b(UNIX_TIMESTAMP|TIMESTAMPDIFF|DATE_ADD|DATE_SUB|EXTRACT|QUARTER)\s*\(",
        "string_function": r"\b(CHAR_LENGTH|LENGTH|CONCAT|TRIM|UPPER|LOWER|SUBSTRING)\s*\(",
        "null_semantics": r"\b(NULL|NOT\s+IN)\b",
    }
    for name, pat in risk_patterns.items():
        if re.search(pat, up):
            risk.append(name)
    return sorted(set(static)), sorted(set(risk))


def process_dataset(path: Path, out_path: Path) -> int:
    rows = []
    for row in read_jsonl(path):
        f1 = features_one(row["sql1"])
        f2 = features_one(row["sql2"])
        features = merge_features(f1, f2, row)
        static_flags, risk_flags = flags_for_pair(row["sql1"], row["sql2"])
        rows.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "features": features,
                "bucket": assign_bucket(features),
                "static_unsupported_flags": static_flags,
                "risk_flags": risk_flags,
            }
        )
    return write_jsonl(out_path, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = standard_paths(cfg)
    outputs = {}
    for name, path in paths.items():
        out = PROJECT_ROOT / "outputs/features" / f"{name}.features.jsonl"
        outputs[name] = {"path": str(out), "rows": process_dataset(path, out)}
    update_manifest(PROJECT_ROOT / "outputs", "extract_features", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
