from __future__ import annotations


def t_norm(name: str, c: float, k: float, s: float = 1.0, x: float = 1.0) -> float:
    key = name.lower().replace("t_", "")
    vals = [max(0.0, min(1.0, v)) for v in (c, k, s, x)]
    if key == "min":
        return min(vals)
    if key == "prod":
        q = 1.0
        for v in vals:
            q *= v
        return q
    if key == "lukasiewicz":
        return max(0.0, sum(vals) - len(vals) + 1)
    raise ValueError(f"Unknown T-norm: {name}")


def apply_tnorm(df, name: str, tau: float):
    out = df.copy()
    out["Q_i"] = [
        t_norm(name, float(r.c_i), float(r.k_i), float(r.s_i), float(r.x_i))
        for r in out.itertuples()
    ]
    out["t_norm"] = name
    out["tau"] = tau
    out["accepted"] = out["Q_i"] >= tau
    return out


def apply_conf_threshold(df, tau: float):
    out = df.copy()
    out["Q_i"] = out["confidence"]
    out["tau"] = tau
    out["accepted"] = out["confidence"] >= tau
    out["t_norm"] = "confidence"
    return out
