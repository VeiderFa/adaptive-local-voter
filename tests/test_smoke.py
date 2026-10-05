"""Fast smoke test: tiny simulation + one small figure.

Run with:  pytest -q
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from avm.avm_two_state_v2 import simulate_jit  # noqa: E402
from avm import plots, sweep  # noqa: E402


def test_simulate_jit_returns_expected_tuple():
    N = 30
    G = nx.erdos_renyi_graph(N, 4 / (N - 1), seed=42)
    adj = nx.to_numpy_array(G)
    opinions = np.array([0] * (N // 2) + [1] * (N // 2))
    np.random.seed(0)
    np.random.shuffle(opinions)

    result = simulate_jit(
        adj, opinions, 2_000, "global", 0.5, False, True,
        track=True, ignore=True, n_steps=50, homo_prob=1.0,
    )
    assert len(result) == 13


def test_quick_sweep_and_plot(tmp_path):
    results = sweep.run_sweep(
        graph_type="ER", out_dir=str(tmp_path), num_agents=30, num_run=2,
        k_val_list=(2, 4), phi_list=(0, 0.5, 1.0), homo_list=(1.0,),
        steps_val=2_000, n_steps=50, n_jobs=1, seed=1,
    )
    assert os.path.exists(tmp_path / "results_global_enhanced_sweep_n30.pkl")

    out = tmp_path / "figs"
    out.mkdir()
    plots.plot_homophily_polarization_echo_chambers(
        results, network_type="ER", k_val=4, homo_val=1.0,
        out_dir=str(out), num_agents=30, n_steps=50, steps_val=2_000,
        phi_samples=(0, 0.5, 1.0),
    )
    assert any(p.name.startswith("homophily_polarization_echo_chambers") for p in out.iterdir())
