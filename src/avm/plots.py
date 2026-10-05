"""Figure generation for the adaptive voter model manuscript.

The plotting code is a direct extraction of the cells in
``notebooks/AVM_Recent_v2.ipynb`` (sibling repository
``Adaptive_Voter_Model``).  Only the surrounding plumbing (argument
passing and output paths) was added; the figure-drawing logic is
unchanged.

Figures produced
----------------
* ``homophily_polarization_echo_chambers_<NET>_n<N>_k<k>_homo<h>.pdf``
  (3x2 panels: homophily H, magnetization |M|, echo chambers EC+1)
* ``degree_distributions_<NET>_n<N>_k<k>_homo<h>_semilog.pdf``
* ``echo_chamber_distribution_nonisolated_<NET>_k<k>_homo<h>.pdf``
* ``combined_heatmaps_largest_component_<NET>_homo<h>_linearscale.pdf``
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def plot_combined_degree_distributions(
    results_global,
    results_local,
    k_val,
    homo_val=0.9,
    network_type="ER",
    phi_values=(0, 0.5, 1.0),
    label_position="top right",
    log_x=False,
    log_y=False,
    phi_list=None,
    num_agents=100,
    out_dir=".",
):
    """Combined degree distributions for global and local rewiring."""
    if phi_list is None:
        phi_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.rc("font", family="serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def get_color_for_phi(phi_val, rewiring_type):
        if phi_val == 0.0:
            return "black"
        elif rewiring_type == "global":
            green_colors = {
                0.1: "#90EE90", 0.2: "#9ACD32", 0.3: "#228B22", 0.4: "#006400",
                0.5: "#00BE00", 0.6: "#9ACD32", 0.7: "#7CFC00", 0.8: "#00FF00",
                0.9: "#ADFF2F", 1.0: "#3B6B53",
            }
            return green_colors.get(phi_val, "#008000")
        else:
            red_colors = {
                0.1: "#FFB6C1", 0.2: "#FF69B4", 0.3: "#FF1493", 0.4: "#DC143C",
                0.5: "#B22222", 0.6: "#FF0000", 0.7: "#8B0000", 0.8: "#CD5C5C",
                0.9: "#F08080", 1.0: "#FF4500",
            }
            return red_colors.get(phi_val, "#B22222")

    markers = ["s", "^", "D", "v", "<", "o", "P", "*", "X", "h"]

    for i, phi_i in enumerate(phi_values):
        if phi_i not in phi_list:
            continue
        final_degree_dist = results_global[k_val][homo_val]["avg_final_degree_dist"][phi_i]
        final_degree_dist_norm = [x / num_agents for x in final_degree_dist]
        final_degree_std = results_global[k_val][homo_val]["std_final_degree_dist"][phi_i]
        final_degree_std_norm = [x / num_agents for x in final_degree_std]
        degrees = np.arange(len(final_degree_dist))
        mask = np.array(final_degree_dist_norm) > 0
        filtered_degrees = degrees[mask]
        filtered_dist = np.array(final_degree_dist_norm)[mask]
        filtered_std = np.array(final_degree_std_norm)[mask]
        if len(filtered_degrees) > 0:
            if log_y:
                rel_error = np.where(filtered_dist > 0, filtered_std / filtered_dist, 0)
                rel_error = np.minimum(rel_error, 0.9)
                y_upper = filtered_dist * (1 + rel_error)
                y_lower = filtered_dist * np.maximum(1 - rel_error, 0.01 * filtered_dist)
                yerr_lower = filtered_dist - y_lower
                yerr_upper = y_upper - filtered_dist
                yerr = [yerr_lower, yerr_upper]
            else:
                yerr = filtered_std
            color = get_color_for_phi(phi_i, "global")
            axes[0].errorbar(
                filtered_degrees, filtered_dist, yerr=yerr,
                marker=markers[i % len(markers)], markersize=6, linestyle="",
                linewidth=2, color=color, alpha=0.8, capsize=3, capthick=1,
                elinewidth=1, label=rf"$p_g = {phi_i}$",
            )

    axes[0].set_xlabel(r"$k_i$", fontsize=20)
    axes[0].set_ylabel(r"$P(k_i)$", fontsize=20)
    axes[0].grid(True, alpha=0.3)
    if log_x:
        axes[0].set_xscale("log")
        axes[0].set_xlim(0.8, num_agents)
    else:
        axes[0].set_xlim(-0.5, 50)
    if log_y:
        axes[0].set_yscale("log")
        axes[0].set_ylim(5e-6, 1.5)
    else:
        axes[0].set_ylim(-0.01, 0.55)
    axes[0].legend(loc="center right", bbox_to_anchor=(0.5, 0.42, 0.5, 0.5))

    for i, phi_i in enumerate(phi_values):
        if phi_i not in phi_list:
            continue
        final_degree_dist = results_local[k_val][homo_val]["avg_final_degree_dist"][phi_i]
        final_degree_dist_norm = [x / num_agents for x in final_degree_dist]
        final_degree_std = results_local[k_val][homo_val]["std_final_degree_dist"][phi_i]
        final_degree_std_norm = [x / num_agents for x in final_degree_std]
        degrees = np.arange(len(final_degree_dist))
        mask = np.array(final_degree_dist_norm) > 0
        filtered_degrees = degrees[mask]
        filtered_dist = np.array(final_degree_dist_norm)[mask]
        filtered_std = np.array(final_degree_std_norm)[mask]
        if len(filtered_degrees) > 0:
            if log_y:
                rel_error = np.where(filtered_dist > 0, filtered_std / filtered_dist, 0)
                rel_error = np.minimum(rel_error, 0.9)
                y_upper = filtered_dist * (1 + rel_error)
                y_lower = filtered_dist * np.maximum(1 - rel_error, 0.01 * filtered_dist)
                yerr_lower = filtered_dist - y_lower
                yerr_upper = y_upper - filtered_dist
                yerr = [yerr_lower, yerr_upper]
            else:
                yerr = filtered_std
            color = get_color_for_phi(phi_i, "local")
            axes[1].errorbar(
                filtered_degrees, filtered_dist, yerr=yerr,
                marker=markers[i % len(markers)], markersize=6, linestyle="",
                linewidth=2, color=color, alpha=0.8, capsize=3, capthick=1,
                elinewidth=1, label=rf"$p_l = {phi_i}$",
            )

    axes[1].set_xlabel(r"$k_i$", fontsize=20)
    axes[1].grid(True, alpha=0.3)
    if log_x:
        axes[1].set_xscale("log")
        axes[1].set_xlim(0.8, num_agents)
    else:
        axes[1].set_xlim(-0.5, 50)
    if log_y:
        axes[1].set_yscale("log")
        axes[1].set_ylim(5e-6, 1.5)
    else:
        axes[1].set_ylim(-0.01, 0.55)
    axes[1].legend(loc="center right", bbox_to_anchor=(0.5, 0.42, 0.5, 0.5))

    network_labels = {"ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK"}
    label_text = network_labels.get(network_type, network_type)
    label_positions = {
        "top left": (0.05, 0.95), "top right": (0.95, 0.95),
        "bottom left": (0.05, 0.05), "bottom right": (0.95, 0.05),
        "center left": (0.05, 0.5), "center right": (0.95, 0.5),
    }
    x_pos, y_pos = label_positions.get(label_position, (0.95, 0.95))
    axes[0].text(
        x_pos, y_pos, label_text, transform=axes[0].transAxes,
        fontsize=20, fontweight="bold",
        ha="right" if "right" in label_position else "left",
        va="top" if "top" in label_position else "bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    scale_suffix = ""
    if log_x and log_y:
        scale_suffix = "loglog"
    elif log_y:
        scale_suffix = "semilog"
    elif log_x:
        scale_suffix = "logx"
    filename = (
        f"degree_distributions_{network_type}_n{num_agents}_k{k_val}"
        f"_homo{homo_val}_{scale_suffix}.pdf"
    )
    plt.savefig(os.path.join(out_dir, filename), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return filename


def plot_homophily_polarization_echo_chambers(
    results,
    network_type="ER",
    k_val=4,
    homo_val=1.0,
    out_dir=".",
    num_agents=100,
    n_steps=100,
    steps_val=1_000_000,
    phi_samples=(0, 0.3, 0.5, 0.7, 1),
):
    """3x2 time-evolution panel: H, |M| and EC+1 for global/local rewiring.

    Direct port of cell 22 of ``AVM_Recent_v2.ipynb``.
    """
    plt.rc("axes", prop_cycle=plt.rcParams["axes.prop_cycle"])
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True)

    add_size = 7
    TICK_SIZE = 15 + add_size
    LABEL_SIZE = 18 + add_size
    LEGEND_SIZE = 10 + add_size
    LINE_WIDTH = 3

    network_labels = {"ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK"}
    label_text = network_labels.get(network_type, network_type)

    cmap = plt.cm.summer
    cmap2 = plt.cm.autumn
    colors = [cmap(i / len(phi_samples)) for i in range(len(phi_samples))]
    colors2 = [cmap2(i / len(phi_samples)) for i in range(len(phi_samples))]

    for phi_i, color in zip(phi_samples, colors):
        data = results["global"][k_val][homo_val]
        avg_homophily = data["avg_ind_homophily_history"][phi_i]
        std_homophily = data["std_ind_homophily_history"][phi_i]
        upper = np.clip(avg_homophily + std_homophily, 0, 1)
        lower = np.clip(avg_homophily - std_homophily, 0, 1)
        time_steps = np.arange(0, len(avg_homophily) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[0, 0].plot(time_steps, avg_homophily, color=plot_color,
                       linewidth=LINE_WIDTH, label=rf"$p_g = {phi_i:.1f}$")
        axs[0, 0].fill_between(time_steps, lower, upper, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[0, 0].set_ylabel(r"$H$", fontsize=LABEL_SIZE)
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].set_xscale("log")
    axs[0, 0].tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    axs[0, 0].legend(frameon=True, framealpha=1, fontsize=LEGEND_SIZE, loc="lower right")

    for phi_i, color in zip(phi_samples, colors2):
        data = results["local"][k_val][homo_val]
        avg_homophily = data["avg_ind_homophily_history"][phi_i]
        std_homophily = data["std_ind_homophily_history"][phi_i]
        upper = np.clip(avg_homophily + std_homophily, 0, 1)
        lower = np.clip(avg_homophily - std_homophily, 0, 1)
        time_steps = np.arange(0, len(avg_homophily) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[0, 1].plot(time_steps, avg_homophily, color=plot_color,
                       linewidth=LINE_WIDTH, label=rf"$p_l = {phi_i:.1f}$")
        axs[0, 1].fill_between(time_steps, lower, upper, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].set_xscale("log")
    axs[0, 1].tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    axs[0, 1].legend(frameon=True, framealpha=1, fontsize=LEGEND_SIZE, loc="lower right")

    for phi_i, color in zip(phi_samples, colors):
        data = results["global"][k_val][homo_val]
        avg_polarization = data["avg_polarization_history"][phi_i]
        std_polarization = data["std_polarization_history"][phi_i]
        upper = np.clip(avg_polarization + std_polarization, 0, 1)
        lower = np.clip(avg_polarization - std_polarization, 0, 1)
        time_steps = np.arange(0, len(avg_polarization) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[1, 0].plot(time_steps, avg_polarization, color=plot_color, linewidth=LINE_WIDTH)
        axs[1, 0].fill_between(time_steps, lower, upper, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[1, 0].set_ylabel(r"$|M|$", fontsize=LABEL_SIZE)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_xscale("log")
    axs[1, 0].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    for phi_i, color in zip(phi_samples, colors2):
        data = results["local"][k_val][homo_val]
        avg_polarization = data["avg_polarization_history"][phi_i]
        std_polarization = data["std_polarization_history"][phi_i]
        upper = np.clip(avg_polarization + std_polarization, 0, 1)
        lower = np.clip(avg_polarization - std_polarization, 0, 1)
        time_steps = np.arange(0, len(avg_polarization) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[1, 1].plot(time_steps, avg_polarization, color=plot_color, linewidth=LINE_WIDTH)
        axs[1, 1].fill_between(time_steps, lower, upper, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_xscale("log")
    axs[1, 1].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    for phi_i, color in zip(phi_samples, colors):
        data = results["global"][k_val][homo_val]
        avg_homogeneous_comp = np.array(data["avg_homogeneous_component_history"][phi_i]) + 1
        std_homogeneous_comp = np.array(data["std_homogeneous_component_history"][phi_i]) + 1
        lower = np.clip(avg_homogeneous_comp - std_homogeneous_comp, 1, None)
        upper = avg_homogeneous_comp + std_homogeneous_comp
        time_steps = np.arange(0, len(avg_homogeneous_comp) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[2, 0].plot(time_steps, avg_homogeneous_comp, color=plot_color, linewidth=LINE_WIDTH)
        axs[2, 0].fill_between(time_steps, lower, upper, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[2, 0].set_ylabel(r"$EC+1$", fontsize=LABEL_SIZE)
    axs[2, 0].set_xlabel(r"$t$", fontsize=LABEL_SIZE)
    axs[2, 0].set_ylim(0.5, 1.05 * num_agents)
    axs[2, 0].set_xscale("log")
    axs[2, 0].set_yscale("log")
    axs[2, 0].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    for phi_i, color in zip(phi_samples, colors2):
        data = results["local"][k_val][homo_val]
        avg_homogeneous_comp = np.array(data["avg_homogeneous_component_history"][phi_i]) + 1
        std_homogeneous_comp = np.array(data["std_homogeneous_component_history"][phi_i]) + 1
        lower = np.clip(avg_homogeneous_comp - std_homogeneous_comp, 1, None)
        upper = avg_homogeneous_comp + std_homogeneous_comp
        time_steps = np.arange(0, len(avg_homogeneous_comp) * n_steps, n_steps)
        plot_color = "black" if phi_i == 0 else color
        axs[2, 1].plot(time_steps, avg_homogeneous_comp, color=plot_color, linewidth=LINE_WIDTH)
        axs[2, 1].fill_between(time_steps, lower, upper + 1, color=plot_color,
                               alpha=0.2, linewidth=0)
    axs[2, 1].set_xlabel(r"$t$", fontsize=LABEL_SIZE)
    axs[2, 1].set_ylim(0.5, 1.05 * num_agents)
    axs[2, 1].set_xscale("log")
    axs[2, 1].set_yscale("log")
    axs[2, 1].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    for ax in axs.flat:
        ax.set_xlim(n_steps, steps_val)
    plt.tight_layout(pad=3.0)
    fig.align_ylabels(axs[:, 0])
    fig.align_ylabels(axs[:, 1])

    filename = (
        f"homophily_polarization_echo_chambers_{network_type}_n{num_agents}"
        f"_k{k_val}_homo{homo_val}.pdf"
    )
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {filename}")
    return filename


def plot_echo_chamber_distribution(
    results_global,
    results_local,
    k_val,
    homo_val,
    output_dir,
    network_type="ER",
    isolated=False,
    label_position="top right",
    num_agents=100,
):
    """Echo-chamber size distribution. Port of cell 25."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    TICK_SIZE = 24
    LABEL_SIZE = 24
    LEGEND_SIZE = 24

    datag = results_global[k_val][homo_val]
    phi_list_full = datag["phi_list"]
    dataglen = len(phi_list_full)
    phi_list = [phi_list_full[0], phi_list_full[2], phi_list_full[int(dataglen / 2)], phi_list_full[-1]]
    colors = ["#044104", "#216E39", "#56FD8B", "#13EE13"]
    colors2 = ["#720303", "#921D1D", "#D46363", "#FFC400"]

    def get_clipped_errors(freqs, errs):
        freqs = np.clip(freqs, 0, 1)
        lower_err = np.minimum(errs, freqs)
        upper_err = np.minimum(errs, 1 - freqs)
        return np.vstack([lower_err, upper_err])

    def process_data(avg_dist, std_dist, num_agents):
        if isolated is False:
            non_isolated_mask = np.arange(len(avg_dist)) >= 1
        else:
            non_isolated_mask = np.arange(len(avg_dist)) >= 0
        total_non_isolated = np.sum(avg_dist[non_isolated_mask])
        if total_non_isolated > 0:
            avg_dist_normalized = np.zeros_like(avg_dist)
            avg_dist_normalized[non_isolated_mask] = avg_dist[non_isolated_mask] / total_non_isolated
            std_dist_normalized = std_dist / total_non_isolated
        else:
            avg_dist_normalized = np.zeros_like(avg_dist)
            std_dist_normalized = std_dist
        nonzero_indices = np.where((avg_dist_normalized > 0) & non_isolated_mask)[0]
        sizes = (nonzero_indices + 1) / num_agents
        freqs = avg_dist_normalized[nonzero_indices]
        errs = std_dist_normalized[nonzero_indices]
        return sizes, freqs, errs

    datag = results_global[k_val][homo_val]
    for i, phi in enumerate(phi_list):
        sizes, freqs, errs = process_data(
            datag["avg_wcc_dist"][phi], datag["std_wcc_dist"][phi], num_agents
        )
        yerr = get_clipped_errors(freqs, errs)
        ax1.errorbar(sizes, freqs, yerr=yerr, fmt="o", markersize=8, capsize=4,
                     color=(colors[i] if phi > 0 else "black"), alpha=0.8,
                     label=rf"$p_g={np.round(phi, 1)}$")

    ax1.set_xlabel(r"$s_i$", fontsize=LABEL_SIZE)
    ax1.set_ylabel(r"$P(s_i)$", fontsize=LABEL_SIZE)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="upper left")
    ax1.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    datal = results_local[k_val][homo_val]
    for i, phi in enumerate(phi_list):
        sizes, freqs, errs = process_data(
            datal["avg_wcc_dist"][phi], datal["std_wcc_dist"][phi], num_agents
        )
        yerr = get_clipped_errors(freqs, errs)
        ax2.errorbar(sizes, freqs, yerr=yerr, fmt="o", markersize=8, capsize=4,
                     color=(colors2[i] if phi > 0 else "black"), alpha=0.8,
                     label=rf"$p_l={np.round(phi, 1)}$")

    ax2.set_xlabel(r"$s_i$", fontsize=LABEL_SIZE)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="upper left")
    ax2.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    network_labels = {"ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK"}
    label_text = network_labels.get(network_type, network_type)
    label_positions = {
        "top left": (0.05, 0.95), "top right": (0.95, 0.95),
        "bottom left": (0.05, 0.05), "bottom right": (0.95, 0.05),
        "center left": (0.05, 0.5), "center right": (0.95, 0.5),
    }
    _, y_pos = label_positions.get(label_position, (0.95, 0.95))
    ax1.text(0.55, y_pos, label_text, transform=ax1.transAxes, fontsize=34,
             fontweight="bold", ha="right" if "right" in label_position else "left",
             va="top" if "top" in label_position else "bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.8, edgecolor="black"))

    plt.tight_layout()
    isolation_suffix = "nonisolated" if not isolated else "all"
    filename = f"echo_chamber_distribution_{isolation_suffix}_{network_type}_k{k_val}_homo{homo_val}.pdf"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"Saved {filename}")
    return filename


def plot_heatmaps(
    results,
    network_type="ER",
    homo_val_heatmap=1.0,
    out_dir=".",
    num_agents=100,
    k_val_list=None,
    phi_list=None,
    use_log_scale_convergence=False,
):
    """2x2 heatmaps of largest-component size s1 and convergence time tau.

    Direct port of cell 27.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), sharex="col", sharey="row")
    plt.subplots_adjust(hspace=0.1, wspace=0.15)

    size_increase = 6
    TICK_SIZE = 20 + size_increase
    LABEL_SIZE = 20 + size_increase
    TITLE_SIZE = 22 + size_increase
    log_scale_min_threshold = 1e-3

    network_labels = {"ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK"}
    label_text = network_labels.get(network_type, network_type)

    k_values = np.array(k_val_list)
    phi_values = np.array(phi_list)
    phi_grid, k_grid = np.meshgrid(phi_values, k_values)

    largest_component_sizes = np.zeros_like(phi_grid)
    for i, k_val in enumerate(k_values):
        for j, phi_val in enumerate(phi_values):
            wcc_dist = results["global"][k_val][homo_val_heatmap]["avg_wcc_dist"][phi_val]
            non_zero_indices = np.nonzero(wcc_dist)[0]
            largest_component_size = (non_zero_indices[-1] + 1) if len(non_zero_indices) > 0 else 0
            largest_component_sizes[i, j] = largest_component_size / num_agents
    im1 = axs[0, 0].pcolormesh(phi_grid, k_grid, largest_component_sizes,
                               cmap="magma", vmin=0, vmax=1, shading="auto")
    axs[0, 0].set_xticks(phi_values)
    axs[0, 0].set_yticks(k_values)
    axs[0, 0].tick_params(axis="both", labelsize=TICK_SIZE)
    axs[0, 0].set_ylabel(r"$k_{avg}$", fontsize=LABEL_SIZE)
    axs[0, 0].text(0.12, 0.95, label_text, transform=axs[0, 0].transAxes,
                   fontsize=28, fontweight="bold", ha="right", va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             alpha=0.9, edgecolor="black"))

    largest_component_sizes = np.zeros_like(phi_grid)
    for i, k_val in enumerate(k_values):
        for j, phi_val in enumerate(phi_values):
            wcc_dist = results["local"][k_val][homo_val_heatmap]["avg_wcc_dist"][phi_val]
            non_zero_indices = np.nonzero(wcc_dist)[0]
            largest_component_size = (non_zero_indices[-1] + 1) if len(non_zero_indices) > 0 else 0
            largest_component_sizes[i, j] = largest_component_size / num_agents
    im2 = axs[0, 1].pcolormesh(phi_grid, k_grid, largest_component_sizes,
                               cmap="magma", vmin=0, vmax=1, shading="auto")
    axs[0, 1].set_xticks(phi_values)
    axs[0, 1].set_yticks(k_values)
    axs[0, 1].tick_params(axis="both", labelsize=TICK_SIZE)
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label(r"$s_1$", fontsize=LABEL_SIZE)
    cbar1.ax.tick_params(labelsize=TICK_SIZE)

    conv_times_global = np.zeros_like(phi_grid)
    for i, k_val in enumerate(k_values):
        conv_times_global[i, :] = results["global"][k_val][homo_val_heatmap]["avg_convergence_times"]
    conv_times_local = np.zeros_like(phi_grid)
    for i, k_val in enumerate(k_values):
        conv_times_local[i, :] = results["local"][k_val][homo_val_heatmap]["avg_convergence_times"]

    all_conv_times = np.concatenate(
        [results["global"][k][homo_val_heatmap]["avg_convergence_times"] for k in k_values]
        + [results["local"][k][homo_val_heatmap]["avg_convergence_times"] for k in k_values]
    )
    valid_conv_times = all_conv_times[all_conv_times > 0]

    if use_log_scale_convergence:
        from matplotlib.colors import LogNorm
        vmin_log = max(log_scale_min_threshold, np.min(valid_conv_times))
        vmax_log = np.max(valid_conv_times)
        conv_times_global_log = np.where(conv_times_global > 0,
                                         np.maximum(conv_times_global, log_scale_min_threshold),
                                         log_scale_min_threshold)
        conv_times_local_log = np.where(conv_times_local > 0,
                                        np.maximum(conv_times_local, log_scale_min_threshold),
                                        log_scale_min_threshold)
        im3 = axs[1, 0].pcolormesh(phi_grid, k_grid, conv_times_global_log,
                                   cmap="viridis", norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
                                   shading="auto")
        im4 = axs[1, 1].pcolormesh(phi_grid, k_grid, conv_times_local_log,
                                   cmap="viridis", norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
                                   shading="auto")
        colorbar_label = r"$\tau$"
    else:
        vmax_linear = np.max(valid_conv_times)
        im3 = axs[1, 0].pcolormesh(phi_grid, k_grid, conv_times_global,
                                   cmap="viridis", vmin=0, vmax=vmax_linear, shading="auto")
        im4 = axs[1, 1].pcolormesh(phi_grid, k_grid, conv_times_local,
                                   cmap="viridis", vmin=0, vmax=vmax_linear, shading="auto")
        colorbar_label = r"$\tau$"

    for ax, xlabel in zip([axs[1, 0], axs[1, 1]], [r"$p_g$", r"$p_l$"]):
        ax.set_xticks(phi_values)
        ax.set_yticks(k_values)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    axs[1, 0].set_ylabel(r"$k_{avg}$", fontsize=LABEL_SIZE)

    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2)
    cbar2.set_label(colorbar_label, fontsize=LABEL_SIZE)
    cbar2.ax.tick_params(labelsize=TICK_SIZE)

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
                fontsize=TITLE_SIZE, fontweight="bold", va="top")

    scale_suffix = "_logscale" if use_log_scale_convergence else "_linearscale"
    filename = (
        f"combined_heatmaps_largest_component_{network_type}"
        f"_homo{homo_val_heatmap}{scale_suffix}.pdf"
    )
    plt.savefig(os.path.join(out_dir, filename), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return filename
