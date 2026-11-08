"""Re-run the offline stages from the committed LLM rankings and check the results.

Stages 07-10 need no API keys or GPUs: they start from the saved rankings
(outputs/rankings/raw_rankings.parquet), the embedding scores and the verified
attributes. This script copies those inputs into a temporary folder, runs the
stages there (the committed outputs are never touched), and compares every table
and report with the committed version.

    python scripts/reproduce.py                    # Stages 07-10 (Stage 10 takes 25-60 min)
    python scripts/reproduce.py --stages 07 08 09  # about 1 minute
    python scripts/reproduce.py --keep             # keep the temporary folder for inspection
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    "07": ("07_ranking_analysis.py", "analysis"),
    "08": ("08_substitution_shares.py", "substitution"),
    "09": ("09_scenario_inputs.py", "scenarios"),
    "10": ("10_assortment_experiment.py", "assortment"),
}
INPUTS = ["src", "data/processed", "outputs/rankings", "outputs/embeddings", "outputs/attributes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--keep", action="store_true", help="keep the temporary working folder")
    args = ap.parse_args()
    stages = sorted(args.stages)

    work = Path(tempfile.mkdtemp(prefix="reproduce_"))
    try:
        for rel in INPUTS:
            shutil.copytree(ROOT / rel, work / rel, ignore=shutil.ignore_patterns("__pycache__"))
        # A later stage reads the outputs of the stage before it. When that earlier
        # stage is not being re-run, start from its committed outputs.
        for code, (_, folder) in STAGES.items():
            if code < stages[0]:
                shutil.copytree(ROOT / "outputs" / folder, work / "outputs" / folder)

        for code in stages:
            script, _ = STAGES[code]
            print(f"\n=== Stage {code}: {script}", flush=True)
            t0 = time.perf_counter()
            subprocess.run([sys.executable, str(work / "src" / "pipeline" / script)], cwd=work, check=True)
            print(f"=== Stage {code} finished in {time.perf_counter() - t0:.0f} s", flush=True)

        print("\n=== Comparing with the committed outputs")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_outputs.py"),
                                 str(ROOT / "outputs"), str(work / "outputs"),
                                 "--stages", *[STAGES[c][1] for c in stages]])
        return result.returncode
    finally:
        if args.keep:
            print(f"\nWorking folder kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
