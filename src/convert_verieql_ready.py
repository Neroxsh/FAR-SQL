#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import PROJECT_ROOT, load_config, parse_json_maybe, read_jsonl, standard_paths, update_manifest, write_jsonl


RE_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\s+`?([A-Za-z0-9_]+)`?\s*\((.*?)\)\s*;", re.I | re.S)
RE_COL_DEF = re.compile(r"`?([A-Za-z0-9_]+)`?\s+([A-Za-z0-9_]+(?:\s*\([^)]*\))?)", re.I)
RE_PK = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.I)
RE_FK = re.compile(r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+`?([A-Za-z0-9_]+)`?\s*\(([^)]*)\)", re.I)
RE_INLINE_REF = re.compile(r"REFERENCES\s+`?([A-Za-z0-9_]+)`?\s*\(([^)]*)\)", re.I)


def norm_name(x: Any) -> str:
    return str(x).strip().strip("`").upper()


def norm_type(x: Any) -> str:
    if x is None:
        return "VARCHAR(200)"
    t = str(x).strip().upper()
    if not t or t == "TEXT":
        return "VARCHAR(200)"
    if t == "BOOL":
        return "BOOLEAN"
    if t.startswith("VARCHAR") and "(" not in t:
        return "VARCHAR(200)"
    return t


def split_cols(block: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    for ch in block:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
    part = "".join(cur).strip()
    if part:
        parts.append(part)
    return parts


def primary(table: str, col: str) -> Dict[str, Any]:
    return {"primary": [{"value": f"{norm_name(table)}__{norm_name(col)}"}]}


def foreign(child_table: str, child_col: str, parent_table: str, parent_col: str) -> Dict[str, Any]:
    return {
        "foreign": [
            {"value": f"{norm_name(child_table)}__{norm_name(child_col)}"},
            {"value": f"{norm_name(parent_table)}__{norm_name(parent_col)}"},
        ]
    }


def is_constraint_key(key: Any) -> bool:
    return str(key).strip().lower() in {"__constraints__", "__constraint__", "constraints", "constraint"}


def constraint_pair(value: Any) -> Tuple[str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str):
        text = value.strip()
        if "." in text:
            table, col = text.split(".", 1)
            return table, col
        if "__" in text:
            table, col = text.split("__", 1)
            return table, col
    if isinstance(value, dict):
        table = value.get("table") or value.get("table_name")
        col = value.get("column") or value.get("column_name") or value.get("col")
        if table is not None and col is not None:
            return str(table), str(col)
    return None


def constraints_from_schema_blob(blob: Any) -> List[Dict[str, Any]]:
    if not isinstance(blob, dict):
        return []
    constraints: List[Dict[str, Any]] = []
    primary_keys = blob.get("primary_keys") or blob.get("PRIMARY_KEYS") or blob.get("primaryKeys") or []
    foreign_keys = blob.get("foreign_keys") or blob.get("FOREIGN_KEYS") or blob.get("foreignKeys") or []

    if isinstance(primary_keys, str):
        primary_keys = parse_json_maybe(primary_keys)
    if isinstance(foreign_keys, str):
        foreign_keys = parse_json_maybe(foreign_keys)

    if isinstance(primary_keys, list):
        for item in primary_keys:
            pair = constraint_pair(item)
            if pair:
                constraints.append(primary(pair[0], pair[1]))

    if isinstance(foreign_keys, list):
        for item in foreign_keys:
            if isinstance(item, dict):
                left = constraint_pair(item.get("from") or item.get("source") or item.get("child"))
                right = constraint_pair(item.get("to") or item.get("target") or item.get("parent"))
                if left and right:
                    constraints.append(foreign(left[0], left[1], right[0], right[1]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                left = constraint_pair(item[0])
                right = constraint_pair(item[1])
                if left and right:
                    constraints.append(foreign(left[0], left[1], right[0], right[1]))
            elif isinstance(item, str) and "=" in item:
                left_text, right_text = [x.strip() for x in item.split("=", 1)]
                left = constraint_pair(left_text)
                right = constraint_pair(right_text)
                if left and right:
                    constraints.append(foreign(left[0], left[1], right[0], right[1]))
    return constraints


def parse_ddl(schema_text: str) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    schema: Dict[str, Dict[str, str]] = {}
    constraints: List[Dict[str, Any]] = []
    text = schema_text
    if not text.strip().endswith(";"):
        text += ";"
    for table, block in RE_CREATE_TABLE.findall(text):
        table_up = norm_name(table)
        cols: Dict[str, str] = {}
        for part in split_cols(block):
            upper = part.upper()
            m_pk = RE_PK.match(part)
            if m_pk:
                for col in m_pk.group(1).split(","):
                    constraints.append(primary(table_up, col))
                continue
            m_fk = RE_FK.match(part)
            if m_fk:
                child_cols = [c.strip() for c in m_fk.group(1).split(",")]
                parent_table = m_fk.group(2)
                parent_cols = [c.strip() for c in m_fk.group(3).split(",")]
                for c, p in zip(child_cols, parent_cols):
                    constraints.append(foreign(table_up, c, parent_table, p))
                continue
            if upper.startswith(("CONSTRAINT", "UNIQUE", "KEY", "INDEX", "CHECK")):
                continue
            m_col = RE_COL_DEF.match(part)
            if not m_col:
                continue
            col, typ = norm_name(m_col.group(1)), norm_type(m_col.group(2))
            cols[col] = typ
            if "PRIMARY KEY" in upper:
                constraints.append(primary(table_up, col))
            m_ref = RE_INLINE_REF.search(part)
            if m_ref:
                constraints.append(foreign(table_up, col, m_ref.group(1), m_ref.group(2).split(",")[0]))
        if cols:
            schema[table_up] = cols
    if not schema:
        raise ValueError("DDL schema did not contain any parseable CREATE TABLE block")
    return schema, constraints


def infer_type(col: str) -> str:
    low = col.lower()
    if low.endswith("_id") or low == "id" or low.endswith("id") or any(k in low for k in ["count", "num", "year", "age", "score", "salary", "price", "capacity"]):
        return "INT"
    if "date" in low or "time" in low:
        return "DATE"
    if low.startswith("is_") or low.startswith("has_"):
        return "BOOLEAN"
    return "VARCHAR(200)"


def parse_spider_schema(obj: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    # Expected shape: {"db_id": {"tables": {"table": {"columns": "*, a, b"}}, ...}}
    if len(obj) == 1 and isinstance(next(iter(obj.values())), dict) and "tables" in next(iter(obj.values())):
        obj = next(iter(obj.values()))
    tables = obj.get("tables", {})
    schema: Dict[str, Dict[str, str]] = {}
    for table, info in tables.items():
        cols_raw = info.get("columns", "") if isinstance(info, dict) else str(info)
        cols = [c.strip() for c in str(cols_raw).split(",") if c.strip() and c.strip() != "*"]
        schema[norm_name(table)] = {norm_name(c): infer_type(c) for c in cols}
    constraints: List[Dict[str, Any]] = []
    for pk in str(obj.get("primary_keys", "")).split(","):
        pk = pk.strip()
        if "." in pk:
            t, c = pk.split(".", 1)
            constraints.append(primary(t, c))
    for fk in str(obj.get("foreign_keys", "")).split(","):
        fk = fk.strip()
        if "=" in fk:
            left, right = [x.strip() for x in fk.split("=", 1)]
            if "." in left and "." in right:
                lt, lc = left.split(".", 1)
                rt, rc = right.split(".", 1)
                constraints.append(foreign(lt, lc, rt, rc))
    if not schema:
        raise ValueError("Spider schema did not contain parseable tables")
    return schema, constraints


def convert_schema(row: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    schema = parse_json_maybe(row.get("schema", {}))
    row_constraint = parse_json_maybe(row.get("constraint"))
    if isinstance(schema, dict):
        if "tables" in schema or (
            len(schema) == 1
            and isinstance(next(iter(schema.values())), dict)
            and "tables" in next(iter(schema.values()))
        ):
            return parse_spider_schema(schema)
        schema_constraints: List[Dict[str, Any]] = []
        table_schema = dict(schema)
        for key in list(table_schema):
            if is_constraint_key(key):
                schema_constraints.extend(constraints_from_schema_blob(table_schema.pop(key)))
        if table_schema and all(isinstance(v, dict) for v in table_schema.values()) and not any(k in table_schema for k in ["tables", "primary_keys", "foreign_keys"]):
            constraints = row_constraint if isinstance(row_constraint, list) else []
            constraints = [*constraints, *schema_constraints]
            return {norm_name(t): {norm_name(c): norm_type(tp) for c, tp in cols.items()} for t, cols in table_schema.items()}, constraints
        return parse_spider_schema(schema)
    if isinstance(schema, str):
        text = schema.strip()
        maybe = parse_json_maybe(text)
        if isinstance(maybe, dict):
            return parse_spider_schema(maybe)
        return parse_ddl(text)
    raise ValueError("Unsupported schema type")


def process_dataset(name: str, path: Path) -> Dict[str, Any]:
    ready_rows = []
    bad_rows = []
    for idx, row in enumerate(read_jsonl(path)):
        try:
            schema, constraints = convert_schema(row)
            ready_rows.append(
                {
                    "index": idx,
                    "id": row["id"],
                    "file": name,
                    "pair": [row["sql1"], row["sql2"]],
                    "sql1": row["sql1"],
                    "sql2": row["sql2"],
                    "schema": schema,
                    "constraint": constraints,
                    "semantic equivalence": row["label"] == "yes",
                    "source_dataset": row["dataset"],
                    "source_name": row.get("source_name", row.get("source_path")),
                    "source_index": row.get("source_index"),
                }
            )
        except Exception as exc:
            bad_rows.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset", name),
                    "label": row.get("label"),
                    "error": str(exc),
                    "normalized_status": "conversion_error",
                }
            )
    ready_path = PROJECT_ROOT / "outputs/verieql_ready" / f"{name}.verieql_ready.jsonl"
    bad_path = PROJECT_ROOT / "outputs/verieql_ready" / f"{name}.conversion_errors.jsonl"
    return {
        "ready_path": str(ready_path),
        "ready_rows": write_jsonl(ready_path, ready_rows),
        "conversion_error_path": str(bad_path),
        "conversion_errors": write_jsonl(bad_path, bad_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = standard_paths(cfg)
    outputs = {}
    for name, path in paths.items():
        if cfg.get("data_sources", {}).get(name, {}).get("format") == "verieql_record":
            outputs[name] = {
                "status": "skipped_precomputed_verieql_record",
                "source": cfg["data_sources"][name]["path"],
            }
            continue
        outputs[name] = process_dataset(name, path)
    update_manifest(PROJECT_ROOT / "outputs", "convert_verieql_ready", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
