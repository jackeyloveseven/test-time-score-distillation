"""CSV row helpers for heterogeneous TPAMI logging."""

from __future__ import annotations


def csv_fieldnames_for_rows(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    common_keys = set(rows[0].keys())
    for row in rows[1:]:
        common_keys &= set(row.keys())
    ordered_common = [key for key in rows[0].keys() if key in common_keys]
    remaining = sorted({key for row in rows for key in row.keys()} - set(ordered_common))
    return ordered_common + remaining


__all__ = ["csv_fieldnames_for_rows"]
