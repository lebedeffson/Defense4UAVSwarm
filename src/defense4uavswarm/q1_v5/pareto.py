from __future__ import annotations

import pandas as pd


def pareto_front(data: pd.DataFrame, maximize: list[str] | None = None, minimize: list[str] | None = None) -> pd.Series:
    maximize = maximize or ["F1"]
    minimize = minimize or ["false_new_tracks"]
    flags: list[bool] = []
    for i, row in data.iterrows():
        dominated = False
        for j, other in data.iterrows():
            if i == j:
                continue
            ge = all(float(other[m]) >= float(row[m]) for m in maximize)
            le = all(float(other[m]) <= float(row[m]) for m in minimize)
            strict = any(float(other[m]) > float(row[m]) for m in maximize) or any(float(other[m]) < float(row[m]) for m in minimize)
            if ge and le and strict:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=data.index)


def dominance_matrix(data: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    rows = []
    methods = sorted(data[method_col].dropna().unique())
    for a in methods:
        for b in methods:
            if a == b:
                continue
            da = data[data[method_col].eq(a)]
            db = data[data[method_col].eq(b)]
            a_dom = _curve_dominates(da, db)
            b_dom = _curve_dominates(db, da)
            rows.append({"method_a": a, "method_b": b, "a_dominates_b": a_dom, "b_dominates_a": b_dom, "curves_cross": not a_dom and not b_dom})
    return pd.DataFrame(rows)


def _curve_dominates(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.empty or b.empty:
        return False
    for _, rb in b.iterrows():
        dominates_point = ((a["F1"] >= rb["F1"]) & (a["false_new_tracks"] <= rb["false_new_tracks"]) & ((a["F1"] > rb["F1"]) | (a["false_new_tracks"] < rb["false_new_tracks"]))).any()
        if not bool(dominates_point):
            return False
    return True

