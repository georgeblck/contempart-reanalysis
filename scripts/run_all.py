"""One-shot driver: validate embeddings, run all stats, rebuild README tables.

Usage:
    uv run python scripts/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"FAILED: {label} (exit {proc.returncode})")
        sys.exit(proc.returncode)
    print(f"  -> {dt:.1f}s")


def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    run(["uv", "run", "python", "-m", "src.step1_link"], "step1 link embeddings")
    run(["uv", "run", "python", "-m", "src.step2_statistics"], "step2 Mantel + PERMANOVA")
    run(["Rscript", "R/dbrda.R"], "step3 db-RDA (R, parallel)")
    run(["uv", "run", "python", "-m", "src.step4_graph"], "step4 social network")
    run(["uv", "run", "python", "scripts/make_readme_tables.py"], "readme tables")

    print("\nAll done.")


if __name__ == "__main__":
    main()
