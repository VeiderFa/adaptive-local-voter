#!/usr/bin/env python3
"""Run the AVM parameter sweep and write ``results_*_enhanced_sweep_n*.pkl``.

By default this reproduces the manuscript grid (N=100; k=2..22; phi=0..1;
homophily 1.0).  It is computationally heavy (see README); use
``--quick`` for a fast smoke run.

Example
-------
    python scripts/run_sweep.py --graph-type ER --out data/Sweep_ER_Undirected_Magnetization
    python scripts/run_sweep.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from avm.sweep import run_sweep  # noqa: E402

MANUSCRIPT_K = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
MANUSCRIPT_PHI = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-type", default="ER", choices=["ER", "WS", "BA", "HK"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-agents", type=int, default=100)
    ap.add_argument("--num-run", type=int, default=100)
    ap.add_argument("--homo-list", type=float, nargs="+", default=[1.0])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional RNG seed; the paper run was unseeded")
    ap.add_argument("--quick", action="store_true",
                    help="small/fast configuration for smoke testing")
    args = ap.parse_args()

    if args.quick:
        kwargs = dict(
            num_agents=30, num_run=2, k_val_list=(2, 4), phi_list=(0, 0.5, 1.0),
            homo_list=(1.0,), steps_val=5_000, n_steps=50, n_jobs=2,
        )
    else:
        kwargs = dict(
            num_agents=args.num_agents, num_run=args.num_run,
            k_val_list=MANUSCRIPT_K, phi_list=MANUSCRIPT_PHI,
            homo_list=tuple(args.homo_list), steps_val=args.steps,
            n_steps=args.n_steps, n_jobs=args.n_jobs,
        )

    run_sweep(graph_type=args.graph_type, out_dir=args.out, seed=args.seed, **kwargs)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
