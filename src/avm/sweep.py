"""Parameter-sweep driver for the adaptive voter model.

Extracted from cells 12/14/16 of ``notebooks/AVM_Recent_v2.ipynb``
(sibling repository ``Adaptive_Voter_Model``).  Simulation logic lives in
:mod:`avm.avm_two_state_v2`; this module only aggregates runs.

NOTE ON RANDOMNESS
------------------
The original pipeline did not set a global seed, so results are
stochastic.  ``seed`` defaults to ``None`` (preserve original behaviour);
pass an integer to make a run reproducible.
"""

from __future__ import annotations

import os
import pickle

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from .avm_two_state_v2 import (
    calculate_global_clustering_coefficient,
    calculate_individual_homophily,
    simulate_jit,
)

GRAPH_TYPES = ("ER", "WS", "BA", "HK")


def build_graph(graph_type, num_agents, k_val):
    """Initialise the network for a given topology and average degree."""
    if graph_type == "WS":
        graph_initial = nx.watts_strogatz_graph(num_agents, k=k_val, p=0.05)
    elif graph_type == "ER":
        p_val = k_val / (num_agents - 1)
        graph_initial = nx.erdos_renyi_graph(num_agents, p=p_val)
    elif graph_type == "BA":
        graph_initial = nx.barabasi_albert_graph(num_agents, m=k_val // 2)
    elif graph_type == "HK":
        graph_initial = nx.powerlaw_cluster_graph(num_agents, m=k_val // 2, p=1)
    else:
        raise ValueError("Unsupported graph type.")
    return graph_initial


def process_single_run_enhanced(
    run_idx,
    phi_i,
    search_val,
    k_val,
    homo_val,
    graph_type="WS",
    num_agents=100,
    steps_val=1_000_000,
    directed_val=False,
    force_val=True,
    ignore=True,
    n_steps=100,
):
    """Run one simulation with comprehensive metric tracking."""
    graph_initial = build_graph(graph_type, num_agents, k_val)
    graph_adj_matrix = nx.to_numpy_array(graph_initial)

    opinions_init = np.array([0] * (num_agents // 2) + [1] * (num_agents // 2))
    np.random.shuffle(opinions_init)

    initial_degrees = np.array([d for n, d in graph_initial.degree()])
    initial_individual_homophily = calculate_individual_homophily(graph_adj_matrix, opinions_init)
    initial_individual_homophily = np.ma.masked_invalid(initial_individual_homophily)
    initial_degree_dist = np.bincount(initial_degrees, minlength=num_agents)
    initial_clustering = calculate_global_clustering_coefficient(graph_adj_matrix)

    result = simulate_jit(
        graph_adj_matrix, opinions_init,
        steps_val, search_val, phi_i, directed_val, force_val,
        track=True, ignore=ignore, n_steps=n_steps, homo_prob=homo_val,
    )

    (magnetization_diff, adj_matrix_final, opinions_final, convergence_time,
     clustering_history, individual_homophily_history, group_homophily_history,
     total_homophily_history, component_history, scc_history, average_opinion_history,
     homogeneous_component_history, polarization_history) = result

    graph_final = nx.from_numpy_array(adj_matrix_final)
    final_degrees = np.array([d for n, d in graph_final.degree()])
    final_degree_dist = np.bincount(final_degrees, minlength=num_agents)
    final_individual_homophily = calculate_individual_homophily(adj_matrix_final, opinions_final)
    final_individual_homophily = np.ma.masked_invalid(final_individual_homophily)
    final_clustering = calculate_global_clustering_coefficient(adj_matrix_final)

    components = list(nx.connected_components(graph_final))
    wcc_sizes = [len(c) for c in components]
    wcc_dist = np.bincount(wcc_sizes, minlength=num_agents + 1)[1:]
    scc_sizes = wcc_sizes
    scc_dist = wcc_dist

    return {
        "convergence_time": convergence_time,
        "converged": convergence_time != -1,
        "magnetization": abs(magnetization_diff),
        "homo_val": homo_val,
        "initial_degree_dist": initial_degree_dist,
        "initial_clustering": initial_clustering,
        "initial_homophily_dist": initial_individual_homophily,
        "final_degree_dist": final_degree_dist,
        "final_clustering": final_clustering,
        "final_homophily_dist": final_individual_homophily,
        "final_polarization": polarization_history[-1] if polarization_history is not None else 0,
        "final_homogeneous_components": homogeneous_component_history[-1] if homogeneous_component_history is not None else 0,
        "wcc_sizes": wcc_sizes,
        "wcc_dist": wcc_dist,
        "scc_sizes": scc_sizes,
        "scc_dist": scc_dist,
        "clustering_history": clustering_history,
        "individual_homophily_history": individual_homophily_history,
        "total_homophily_history": total_homophily_history,
        "component_history": component_history,
        "homogeneous_component_history": homogeneous_component_history,
        "polarization_history": polarization_history,
        "average_opinion_history": average_opinion_history,
    }


def process_sweep_aggregation(
    phi_i, homo_i, search_val, num_run, k_val,
    graph_type="WS", num_agents=100, steps_val=1_000_000,
    directed_val=False, force_val=True, ignore=True, n_steps=100,
):
    """Aggregate ``num_run`` realisations for one (phi, homo, search, k)."""
    print(f"Running simulation for phi={phi_i}, homo_prop={homo_i}, k={k_val}")
    run_results = [
        process_single_run_enhanced(
            run_idx, phi_i, search_val, k_val, homo_i, graph_type,
            num_agents, steps_val, directed_val, force_val, ignore, n_steps,
        )
        for run_idx in range(num_run)
    ]

    n_converged = sum(r["converged"] for r in run_results)
    converged_times = [r["convergence_time"] for r in run_results if r["converged"]]

    stats = {
        "phi_i": phi_i,
        "homo_i": homo_i,
        "k_val": k_val,
        "convergence_probability": n_converged / num_run,
        "avg_convergence_time": np.mean(converged_times) if converged_times else -1,
        "std_convergence_time": np.std(converged_times) if converged_times else 0,
        "avg_convergence_time_all": np.mean([r["convergence_time"] for r in run_results if r["convergence_time"] != -1]),
        "avg_magnetization": np.mean([r["magnetization"] for r in run_results]),
        "std_magnetization": np.std([r["magnetization"] for r in run_results]),
        "avg_initial_clustering": np.mean([r["initial_clustering"] for r in run_results]),
        "std_initial_clustering": np.std([r["initial_clustering"] for r in run_results]),
        "avg_final_clustering": np.mean([r["final_clustering"] for r in run_results]),
        "std_final_clustering": np.std([r["final_clustering"] for r in run_results]),
        "avg_final_polarization": np.mean([r["final_polarization"] for r in run_results]),
        "std_final_polarization": np.std([r["final_polarization"] for r in run_results]),
        "avg_final_homogeneous_components": np.mean([r["final_homogeneous_components"] for r in run_results]),
        "std_final_homogeneous_components": np.std([r["final_homogeneous_components"] for r in run_results]),
    }

    initial_degree_dists = np.array([r["initial_degree_dist"] for r in run_results])
    stats["avg_initial_degree_dist"] = np.mean(initial_degree_dists, axis=0)
    stats["std_initial_degree_dist"] = np.std(initial_degree_dists, axis=0)
    final_degree_dists = np.array([r["final_degree_dist"] for r in run_results])
    stats["avg_final_degree_dist"] = np.mean(final_degree_dists, axis=0)
    stats["std_final_degree_dist"] = np.std(final_degree_dists, axis=0)
    initial_homophily_dists = np.array([r["initial_homophily_dist"] for r in run_results])
    stats["avg_initial_homophily_dist"] = np.mean(initial_homophily_dists, axis=0)
    stats["std_initial_homophily_dist"] = np.std(initial_homophily_dists, axis=0)
    final_homophily_dists = np.array([r["final_homophily_dist"] for r in run_results])
    stats["avg_final_homophily_dist"] = np.mean(final_homophily_dists, axis=0)
    stats["std_final_homophily_dist"] = np.std(final_homophily_dists, axis=0)

    wcc_dists = np.array([r["wcc_dist"] for r in run_results])
    stats["avg_wcc_dist"] = np.mean(wcc_dists, axis=0)
    stats["std_wcc_dist"] = np.std(wcc_dists, axis=0)
    scc_dists = np.array([r["scc_dist"] for r in run_results])
    stats["avg_scc_dist"] = np.mean(scc_dists, axis=0)
    stats["std_scc_dist"] = np.std(scc_dists, axis=0)

    # map internal history names to the keys used by the plotting code
    history_keys = {
        "clustering_history": "clustering_history",
        "individual_homophily_history": "ind_homophily_history",
        "total_homophily_history": "total_homophily_history",
        "component_history": "component_history",
        "homogeneous_component_history": "homogeneous_component_history",
        "polarization_history": "polarization_history",
        "average_opinion_history": "opinion_history",
    }
    for src, dst in history_keys.items():
        if run_results[0][src] is not None:
            min_length = min(len(r[src]) for r in run_results)
            stats["avg_" + dst] = np.mean([r[src][:min_length] for r in run_results], axis=0)
            stats["std_" + dst] = np.std([r[src][:min_length] for r in run_results], axis=0)

    return stats


def run_sweep(
    graph_type="WS",
    out_dir=".",
    num_agents=100,
    num_run=100,
    k_val_list=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22),
    phi_list=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    homo_list=(1.0,),
    steps_val=1_000_000,
    n_steps=100,
    directed_val=False,
    force_val=True,
    ignore=True,
    search_val_list=("global", "local"),
    n_jobs=5,
    seed=None,
):
    """Run the full parameter sweep and write per-strategy ``.pkl`` files."""
    if seed is not None:
        np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    results_enhanced = {}

    for search_val in search_val_list:
        results_enhanced[search_val] = {}
        for k_val in k_val_list:
            print(f"\n{'=' * 60}")
            print(f"Comprehensive sweep: search={search_val}, k={k_val}")
            print(f"{'=' * 60}")
            param_combinations = [(phi_i, homo_i) for phi_i in phi_list for homo_i in homo_list]
            sweep_results = Parallel(n_jobs=n_jobs)(
                delayed(process_sweep_aggregation)(
                    phi_i, homo_i, search_val, num_run, k_val, graph_type,
                    num_agents, steps_val, directed_val, force_val, ignore, n_steps,
                )
                for phi_i, homo_i in param_combinations
            )

            results_enhanced[search_val][k_val] = {}
            for homo_i in homo_list:
                homo_results = [r for r in sweep_results if r["homo_i"] == homo_i]
                results_enhanced[search_val][k_val][homo_i] = {
                    "phi_list": list(phi_list),
                    "homo_val": homo_i,
                    "convergence_probability": np.array([r["convergence_probability"] for r in homo_results]),
                    "avg_convergence_times": np.array([r["avg_convergence_time"] for r in homo_results]),
                    "std_convergence_times": np.array([r["std_convergence_time"] for r in homo_results]),
                    "avg_convergence_times_all": np.array([r["avg_convergence_time_all"] for r in homo_results]),
                    "avg_magnetization": np.array([r["avg_magnetization"] for r in homo_results]),
                    "std_magnetization": np.array([r["std_magnetization"] for r in homo_results]),
                    "avg_initial_clustering": np.array([r["avg_initial_clustering"] for r in homo_results]),
                    "std_initial_clustering": np.array([r["std_initial_clustering"] for r in homo_results]),
                    "avg_final_clustering": np.array([r["avg_final_clustering"] for r in homo_results]),
                    "std_final_clustering": np.array([r["std_final_clustering"] for r in homo_results]),
                    "avg_final_polarization": np.array([r["avg_final_polarization"] for r in homo_results]),
                    "std_final_polarization": np.array([r["std_final_polarization"] for r in homo_results]),
                    "avg_final_homogeneous_components": np.array([r["avg_final_homogeneous_components"] for r in homo_results]),
                    "std_final_homogeneous_components": np.array([r["std_final_homogeneous_components"] for r in homo_results]),
                    "avg_wcc_dist": {r["phi_i"]: r["avg_wcc_dist"] for r in homo_results},
                    "std_wcc_dist": {r["phi_i"]: r["std_wcc_dist"] for r in homo_results},
                    "avg_scc_dist": {r["phi_i"]: r["avg_scc_dist"] for r in homo_results},
                    "std_scc_dist": {r["phi_i"]: r["std_scc_dist"] for r in homo_results},
                    "avg_clustering_history": {r["phi_i"]: r["avg_clustering_history"] for r in homo_results},
                    "std_clustering_history": {r["phi_i"]: r["std_clustering_history"] for r in homo_results},
                    "avg_ind_homophily_history": {r["phi_i"]: r["avg_ind_homophily_history"] for r in homo_results},
                    "std_ind_homophily_history": {r["phi_i"]: r["std_ind_homophily_history"] for r in homo_results},
                    "avg_total_homophily_history": {r["phi_i"]: r["avg_total_homophily_history"] for r in homo_results},
                    "std_total_homophily_history": {r["phi_i"]: r["std_total_homophily_history"] for r in homo_results},
                    "avg_component_history": {r["phi_i"]: r["avg_component_history"] for r in homo_results},
                    "std_component_history": {r["phi_i"]: r["std_component_history"] for r in homo_results},
                    "avg_homogeneous_component_history": {r["phi_i"]: r["avg_homogeneous_component_history"] for r in homo_results},
                    "std_homogeneous_component_history": {r["phi_i"]: r["std_homogeneous_component_history"] for r in homo_results},
                    "avg_polarization_history": {r["phi_i"]: r["avg_polarization_history"] for r in homo_results},
                    "std_polarization_history": {r["phi_i"]: r["std_polarization_history"] for r in homo_results},
                    "avg_opinion_history": {r["phi_i"]: r["avg_opinion_history"] for r in homo_results},
                    "std_opinion_history": {r["phi_i"]: r["std_opinion_history"] for r in homo_results},
                    "avg_initial_degree_dist": {r["phi_i"]: r["avg_initial_degree_dist"] for r in homo_results},
                    "std_initial_degree_dist": {r["phi_i"]: r["std_initial_degree_dist"] for r in homo_results},
                    "avg_final_degree_dist": {r["phi_i"]: r["avg_final_degree_dist"] for r in homo_results},
                    "std_final_degree_dist": {r["phi_i"]: r["std_final_degree_dist"] for r in homo_results},
                    "avg_initial_homophily_dist": {r["phi_i"]: r["avg_initial_homophily_dist"] for r in homo_results},
                    "std_initial_homophily_dist": {r["phi_i"]: r["std_initial_homophily_dist"] for r in homo_results},
                    "avg_final_homophily_dist": {r["phi_i"]: r["avg_final_homophily_dist"] for r in homo_results},
                    "std_final_homophily_dist": {r["phi_i"]: r["std_final_homophily_dist"] for r in homo_results},
                }

        results_filename = os.path.join(out_dir, f"results_{search_val}_enhanced_sweep_n{num_agents}.pkl")
        with open(results_filename, "wb") as f:
            pickle.dump(results_enhanced[search_val], f)
        print(f"Saved {results_filename}")

    return results_enhanced
