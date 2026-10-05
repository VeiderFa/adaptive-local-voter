#!/usr/bin/env python3
"""Download the precomputed sweep results from Zenodo.

The manuscript figures are generated from precomputed ``.pkl`` files rather
than re-running the (very expensive) simulations.  The files are archived on
Zenodo with a persistent DOI.

    DOI (to be assigned): 10.5281/zenodo.XXXXXXX

Usage
-----
    # Show the configured DOI and target directory
    python scripts/download_data.py --info

    # Download (requires the DOI to be set in this file)
    python scripts/download_data.py --out data
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ZENODO_DOI = "10.5281/zenodo.XXXXXXX"  # TODO: replace after minting the DOI

# Files to fetch and where they are expected to live after download.
DATASETS = {
    "ER": "Sweep_ER_Undirected_Magnetization",
    "WS": "Sweep_WS_Undirected_Magnetization",
    "BA": "Sweep_BA_Undirected_Magnetization",
}


def _record_url(doi, filename):
    # Zenodo record files are served from /record/<id>/files/<name>
    record = doi.rsplit(".", 1)[-1]
    return f"https://zenodo.org/records/{record}/files/{filename}?download=1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--network", choices=list(DATASETS), default=None)
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    if args.info or "XXXXXXX" in ZENODO_DOI:
        print(f"Zenodo DOI: {ZENODO_DOI}")
        print("Target datasets:")
        for net, folder in DATASETS.items():
            print(f"  {net}: data/{folder}/results_{{global,local}}_enhanced_sweep_n100.pkl")
        if "XXXXXXX" in ZENODO_DOI:
            print(
                "\nThe DOI is not set yet. After publishing the Zenodo deposit, "
                "replace ZENODO_DOI in scripts/download_data.py and re-run:\n"
                "  python scripts/download_data.py --out data"
            )
            return 1
        return 0

    nets = [args.network] if args.network else list(DATASETS)
    os.makedirs(args.out, exist_ok=True)
    for net in nets:
        dest = os.path.join(args.out, DATASETS[net])
        os.makedirs(dest, exist_ok=True)
        for strategy in ("global", "local"):
            name = f"results_{strategy}_enhanced_sweep_n100.pkl"
            url = _record_url(ZENODO_DOI, name)
            out = os.path.join(dest, name)
            print(f"Downloading {url}\n  -> {out}")
            urllib.request.urlretrieve(url, out)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
