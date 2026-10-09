#!/usr/bin/env python3
"""Regenerate the manuscript figures from precomputed sweep results.

Example
-------
    python scripts/make_figures.py --data-dir data/Sweep_ER_Undirected_Magnetization \
        --network ER --figures all --out figures

The data directory must contain the two files produced by
``scripts/run_sweep.py``::

    results_global_enhanced_sweep_n100.pkl
    results_local_enhanced_sweep_n100.pkl

Figure families
---------------
homophily   -> homophily_polarization_echo_chambers_<NET>_n100_k<k>_homo<h>.pdf
degree      -> degree_distributions_<NET>_n100_k<k>_homo<h>_semilog.pdf
echo        -> echo_chamber_distribution_nonisolated_<NET>_k<k>_homo<h>.pdf
heatmaps    -> combined_heatmaps_largest_component_<NET>_homo<h>_linearscale.pdf
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from avm import plots  # noqa: E402

# Configurations used to produce the manuscript (network -> list of dicts).
MANUSCRIPT_FIGURES = {
    "ER": {
        "homophily": [dict(k_val=4, homo_val=1.0)],
        "degree": [dict(k_val=4, homo_val=1.0)],
        "echo": [dict(k_val=4, homo_val=1.0)],
        "heatmaps": [dict(homo_val=1.0)],
    },
    "WS": {
        "homophily": [
            dict(k_val=4, homo_val=1.0),
            dict(k_val=6, homo_val=1.0),
            dict(k_val=8, homo_val=1.0),
            dict(k_val=4, homo_val=0.9),
            dict(k_val=4, homo_val=0.7),
            dict(k_val=4, homo_val=0.5),
        ],
        "degree": [dict(k_val=4, homo_val=1.0)],
        "echo": [dict(k_val=4, homo_val=1.0)],
        "heatmaps": [dict(homo_val=1.0)],
    },
    "BA": {
        "homophily": [dict(k_val=4, homo_val=1.0)],
        "degree": [dict(k_val=4, homo_val=1.0)],
        "echo": [dict(k_val=4, homo_val=1.0)],
        "heatmaps": [dict(homo_val=1.0)],
    },
}


def load_results(data_dir):
    def _load(name):
        with open(os.path.join(data_dir, name), "rb") as f:
            return pickle.load(f)

    return (
        _load("results_global_enhanced_sweep_n100.pkl"),
        _load("results_local_enhanced_sweep_n100.pkl"),
    )


def infer_grid(results_global):
    k_list = list(results_global.keys())
    first_k = k_list[0]
    homo_list = list(results_global[first_k].keys())
    phi_list = list(results_global[first_k][homo_list[0]]["phi_list"])
    return k_list, homo_list, phi_list


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--network", required=True, choices=["ER", "WS", "BA"])
    ap.add_argument("--figures", default="all",
                    help="comma-separated subset of homophily,degree,echo,heatmaps,all")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--num-agents", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--heatmap-start-letter", type=int, default=0,
                    help="offset for heatmap panel letters (0 -> (a)-(d), 4 -> (e)-(h))")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_global, results_local = load_results(args.data_dir)
    k_list, homo_list, phi_list = infer_grid(results_global)
    results = {"global": results_global, "local": results_local}

    requested = (
        ["homophily", "degree", "echo", "heatmaps"]
        if args.figures == "all"
        else [x.strip() for x in args.figures.split(",")]
    )

    # For the heatmap the full k-range and phi-range are required.
    for fam in requested:
        for cfg in MANUSCRIPT_FIGURES[args.network][fam]:
            if fam == "homophily":
                plots.plot_homophily_polarization_echo_chambers(
                    results, network_type=args.network, out_dir=args.out,
                    num_agents=args.num_agents, n_steps=args.n_steps,
                    steps_val=args.steps, **cfg,
                )
            elif fam == "degree":
                plots.plot_combined_degree_distributions(
                    results_global, results_local, phi_list=phi_list,
                    num_agents=args.num_agents, out_dir=args.out,
                    network_type=args.network, phi_values=[0.0, 0.3, 0.5, 1.0],
                    log_x=False, log_y=True, **cfg,
                )
            elif fam == "echo":
                plots.plot_echo_chamber_distribution(
                    results_global, results_local, output_dir=args.out,
                    network_type=args.network, num_agents=args.num_agents,
                    isolated=False, **cfg,
                )
            elif fam == "heatmaps":
                plots.plot_heatmaps(
                    results, network_type=args.network, out_dir=args.out,
                    num_agents=args.num_agents, k_val_list=k_list,
                    phi_list=phi_list, use_log_scale_convergence=False,
                    homo_val_heatmap=cfg["homo_val"],
                    start_letter=args.heatmap_start_letter,
                )
            else:
                raise SystemExit(f"unknown figure family: {fam}")

    print(f"\nFigures written to {args.out}")


if __name__ == "__main__":
    main()
