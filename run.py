#!/usr/bin/env python3
"""Legacy training entry point. Prefer: python scripts/train.py --config configs/....yaml"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "fpt_combined_pn.yaml"

if __name__ == "__main__":
    print(
        "Note: run.py is deprecated. Use:\n"
        "  python scripts/train.py --config configs/<experiment>.yaml\n"
    )
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")
