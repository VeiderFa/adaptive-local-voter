#!/usr/bin/env python3
"""Download the precomputed sweep results from Zenodo.

The manuscript figures are generated from precomputed ``.pkl`` files rather
than re-running the (very expensive) simulations.  The files are archived on
Zenodo with a persistent DOI.

    DOI (to be assigned): 10.5281/zenodo.XXXXXXX

Filename mapping
----------------
The six canonical files share the same local basename
``results_{global,local}_enhanced_sweep_n100.pkl``.  So that all six can live
in a single Zenodo record, they are uploaded with a network prefix::

    Zenodo filename                         Local path
    ER_results_global_enhanced_sweep_n100.pkl  ->  data/Sweep_ER_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl
    ER_results_local_enhanced_sweep_n100.pkl   ->  data/Sweep_ER_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl
    WS_results_global_enhanced_sweep_n100.pkl  ->  data/Sweep_WS_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl
    WS_results_local_enhanced_sweep_n100.pkl   ->  data/Sweep_WS_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl
    BA_results_global_enhanced_sweep_n100.pkl  ->  data/Sweep_BA_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl
    BA_results_local_enhanced_sweep_n100.pkl   ->  data/Sweep_BA_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl

Usage
-----
    # Show the DOI, the naming map and exit (no network access)
    python scripts/download_data.py --info

    # Download all six files (requires the DOI to be set in this file)
    python scripts/download_data.py --out data

    # Download one topology only
    python scripts/download_data.py --out data --network ER
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# Set this after the Zenodo deposit is published.
ZENODO_DOI = "10.5281/zenodo.XXXXXXX"  # TODO: replace after minting the DOI

NETWORKS = ("ER", "WS", "BA")
STRATEGIES = ("global", "local")


def local_name(strategy: str) -> str:
    """Canonical on-disk filename (network folder holds the topology)."""
    return f"results_{strategy}_enhanced_sweep_n100.pkl"


def remote_name(network: str, strategy: str) -> str:
    """Network-prefixed filename as uploaded to Zenodo."""
    return f"{network}_{local_name(strategy)}"


def local_dir(out_root: str, network: str) -> str:
    return os.path.join(out_root, f"Sweep_{network}_Undirected_Magnetization")


def record_url(doi: str, filename: str) -> str:
    record = doi.rsplit(".", 1)[-1]
    return f"https://zenodo.org/records/{record}/files/{filename}?download=1"


def _print_info() -> None:
    print(f"Zenodo DOI: {ZENODO_DOI}")
    print("\nNaming map (Zenodo filename -> local path):")
    for net in NETWORKS:
        for strat in STRATEGIES:
            print(f"  {remote_name(net, strat)}")
            print(f"    -> {os.path.join(local_dir('data', net), local_name(strat))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--network", choices=list(NETWORKS), default=None)
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    if args.info or "XXXXXXX" in ZENODO_DOI:
        _print_info()
        if "XXXXXXX" in ZENODO_DOI:
            print(
                "\nThe DOI is not set yet. After publishing the Zenodo deposit, "
                "replace ZENODO_DOI in scripts/download_data.py and re-run:\n"
                "  python scripts/download_data.py --out data"
            )
            return 1
        return 0

    networks = [args.network] if args.network else list(NETWORKS)
    for net in networks:
        dest = local_dir(args.out, net)
        os.makedirs(dest, exist_ok=True)
        for strat in STRATEGIES:
            url = record_url(ZENODO_DOI, remote_name(net, strat))
            out = os.path.join(dest, local_name(strat))
            print(f"Downloading {url}\n  -> {out}")
            urllib.request.urlretrieve(url, out)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
