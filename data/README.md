# Data

This directory holds the precomputed Monte-Carlo sweep results that the
manuscript figures are generated from. The `.pkl` files are **not stored in
git** (they are 60–250 MB each); they are archived on Zenodo.

## Fetch

```bash
python scripts/download_data.py --out data
```

The DOI is configured in `scripts/download_data.py` (placeholder until the
deposit is published).

## Expected layout

```
data/
  Sweep_ER_Undirected_Magnetization/
    results_global_enhanced_sweep_n100.pkl
    results_local_enhanced_sweep_n100.pkl
  Sweep_WS_Undirected_Magnetization/
    results_global_enhanced_sweep_n100.pkl
    results_local_enhanced_sweep_n100.pkl
  Sweep_BA_Undirected_Magnetization/
    results_global_enhanced_sweep_n100.pkl
    results_local_enhanced_sweep_n100.pkl
```

## Zenodo filenames

All six files share the local basename
`results_{global,local}_enhanced_sweep_n100.pkl`, so on Zenodo they are stored
with a **network prefix** and downloaded back to the canonical local path:

| Zenodo filename | Local path |
|---|---|
| `ER_results_global_enhanced_sweep_n100.pkl` | `data/Sweep_ER_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl` |
| `ER_results_local_enhanced_sweep_n100.pkl` | `data/Sweep_ER_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl` |
| `WS_results_global_enhanced_sweep_n100.pkl` | `data/Sweep_WS_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl` |
| `WS_results_local_enhanced_sweep_n100.pkl` | `data/Sweep_WS_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl` |
| `BA_results_global_enhanced_sweep_n100.pkl` | `data/Sweep_BA_Undirected_Magnetization/results_global_enhanced_sweep_n100.pkl` |
| `BA_results_local_enhanced_sweep_n100.pkl` | `data/Sweep_BA_Undirected_Magnetization/results_local_enhanced_sweep_n100.pkl` |

The script maps these automatically; run `python scripts/download_data.py --info`
to print the map. The canonical `.pkl` files are **not renamed** in place.

## pkl structure

Each file is a dict keyed by average degree `k` (2, 4, 6, …, 22), then by
homophily probability `homo`, then containing aggregated statistics and
per-`phi` histories:

```
results[search][k][homo] = {
    "phi_list": [...],
    "avg_wcc_dist": {phi: ndarray},          # component size distribution
    "std_wcc_dist": {phi: ndarray},
    "avg_ind_homophily_history": {phi: ndarray},
    "avg_polarization_history": {phi: ndarray},   # |M|
    "avg_homogeneous_component_history": {phi: ndarray},  # echo chambers
    "avg_final_degree_dist": {phi: ndarray},
    "avg_convergence_times": ndarray,
    ...
}
```

`search` is `"global"` or `"local"`. `avg_polarization_history` stores the
absolute magnetization `|M| = |N1 - N0| / N`.

## Regenerate

To rebuild the `.pkl` files from scratch (several hours per topology):

```bash
make sweep-ER
make sweep-WS
make sweep-BA
```
