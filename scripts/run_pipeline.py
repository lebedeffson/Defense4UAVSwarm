#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.config import load_config
from defense4uavswarm.pipeline import run_s0_to_s2
from defense4uavswarm.smoke import run_smoke_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stage", choices=["s0_s2"], default="s0_s2")
    p.add_argument("--scenario", choices=["s0", "s1", "s2", "s0_s2"], default="s0_s2")
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--limit-sequences", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        run_smoke_test(cfg)
        return
    if args.stage == "s0_s2":
        run_s0_to_s2(cfg, scenario=args.scenario, eps=args.eps, limit_sequences=args.limit_sequences)


if __name__ == "__main__":
    main()
