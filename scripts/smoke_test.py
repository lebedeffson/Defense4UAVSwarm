#!/usr/bin/env python
from __future__ import annotations

from defense4uavswarm.config import load_config
from defense4uavswarm.smoke import run_smoke_test


if __name__ == "__main__":
    run_smoke_test(load_config("configs/default.yaml"))
