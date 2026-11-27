"""Module 2 - fit MuJoCo parameters to a run recorded by scripts/sysid/collect.py (CMA-ES).

Fits per joint: armature, viscous damping, Coulomb friction (frictionloss), position bias; plus one
command delay. Validation on the held-out multisine. Guide: docs/sysid.md.

Usage
    .venv/bin/python scripts/sysid/fit.py outputs/sysid/<run>                  # full fit (~10-30 min, all cores)
    .venv/bin/python scripts/sysid/fit.py outputs/sysid/<run> --gens 60        # quick look
    .venv/bin/python scripts/sysid/fit.py outputs/sysid/<run> --eval <sim_params.json>   # score given params

Output: <run>/fit_<time>/  sim_params.json (the artifact)  fit_report.txt  fit_*.png
Use it:  SimParams.load(".../sim_params.json").apply_to(model)    # eaiproject/sysid/model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject.sysid.fit import fit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory from collect.py (contains log.npz, meta.json)")
    ap.add_argument("--gens", type=int, default=300, help="max CMA-ES generations")
    ap.add_argument("--pop", type=int, default=32, help="population size")
    ap.add_argument("--threads", type=int, default=None, help="rollout threads (default: all cores)")
    ap.add_argument("--sigma", type=float, default=0.3, help="initial CMA-ES step in the normalised box")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=40, help="stop after this many generations without progress")
    ap.add_argument("--allow-partial", action="store_true", help="also use segments cut short by an abort")
    ap.add_argument("--eval", default=None, metavar="PARAMS_JSON", help="only evaluate these parameters")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fit(a.run, a.out, gens=a.gens, pop=a.pop, threads=a.threads, sigma=a.sigma, seed=a.seed,
        allow_partial=a.allow_partial, patience=a.patience, eval_params=a.eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
