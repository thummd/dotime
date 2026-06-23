"""Visualization utilities for CausalTimePrior."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from causaltimeprior.interventions import InterventionSpec
from causaltimeprior.temporal_graph import TemporalDAG


def plot_paired_timeseries(
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    intervention: InterventionSpec,
    var_idx: int | None = None,
    save_path: str | None = None,
):
    """Plot paired observational and interventional time series.

    Parameters
    ----------
    X_obs : torch.Tensor
        Observational time series of shape (T, N).
    X_int : torch.Tensor
        Interventional time series of shape (T, N).
    intervention : InterventionSpec
        Intervention specification.
    var_idx : int, optional
        Variable index to plot. If None, plots the first intervened variable.
    save_path : str, optional
        Path to save the figure.
    """
    if var_idx is None:
        var_idx = intervention.targets[0] if len(intervention.targets) > 0 else 0

    T = X_obs.shape[0]
    t = np.arange(T)

    _fig, ax = plt.subplots(figsize=(12, 4))

    # Plot observational data
    ax.plot(t, X_obs[:, var_idx].numpy(), label="Observational", color="blue", alpha=0.7)

    # Plot interventional data
    ax.plot(t, X_int[:, var_idx].numpy(), label="Interventional", color="red", alpha=0.7)

    # Highlight intervention period
    if len(intervention.times) > 0:
        int_start = min(intervention.times)
        int_end = max(intervention.times)
        ax.axvspan(int_start, int_end, alpha=0.2, color="yellow", label="Intervention Period")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Variable {var_idx}")
    ax.set_title(f"Paired Time Series (Intervention: {intervention.intervention_type.value})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_intervention_effect(
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    intervention: InterventionSpec,
    var_idx: int | None = None,
    save_path: str | None = None,
):
    """Plot the causal effect of intervention (difference between int and obs).

    Parameters
    ----------
    X_obs : torch.Tensor
        Observational time series of shape (T, N).
    X_int : torch.Tensor
        Interventional time series of shape (T, N).
    intervention : InterventionSpec
        Intervention specification.
    var_idx : int, optional
        Variable index to plot. If None, plots the first intervened variable.
    save_path : str, optional
        Path to save the figure.
    """
    if var_idx is None:
        var_idx = intervention.targets[0] if len(intervention.targets) > 0 else 0

    T = X_obs.shape[0]
    t = np.arange(T)

    # Compute causal effect
    causal_effect = (X_int[:, var_idx] - X_obs[:, var_idx]).numpy()

    _fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, causal_effect, color="green", linewidth=2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Highlight intervention period
    if len(intervention.times) > 0:
        int_start = min(intervention.times)
        int_end = max(intervention.times)
        ax.axvspan(int_start, int_end, alpha=0.2, color="yellow", label="Intervention Period")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Causal Effect (Var {var_idx})")
    ax.set_title("Interventional Effect Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_temporal_dag(
    dag: TemporalDAG,
    save_path: str | None = None,
):
    """Visualize the temporal DAG structure.

    Parameters
    ----------
    dag : TemporalDAG
        Temporal DAG to visualize.
    save_path : str, optional
        Path to save the figure.
    """
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot instantaneous DAG (G_0)
    ax1 = axes[0]
    pos = nx.spring_layout(dag.G_0, seed=42)
    nx.draw(
        dag.G_0,
        pos,
        ax=ax1,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        arrows=True,
        arrowsize=20,
    )
    ax1.set_title("Instantaneous Edges (G_0)")

    # Plot lagged edge statistics
    ax2 = axes[1]
    lag_edges = [np.sum(G_k) for G_k in dag.G_lags]
    lags = np.arange(1, dag.K + 1)
    ax2.bar(lags, lag_edges, color="coral")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Number of Edges")
    ax2.set_title("Lagged Edges per Lag")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_all_variables(
    X_obs: torch.Tensor,
    X_int: torch.Tensor,
    intervention: InterventionSpec,
    save_path: str | None = None,
):
    """Plot all variables side by side.

    Parameters
    ----------
    X_obs : torch.Tensor
        Observational time series of shape (T, N).
    X_int : torch.Tensor
        Interventional time series of shape (T, N).
    intervention : InterventionSpec
        Intervention specification.
    save_path : str, optional
        Path to save the figure.
    """
    T, N = X_obs.shape
    t = np.arange(T)

    n_cols = min(3, N)
    n_rows = (N + n_cols - 1) // n_cols

    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten() if N > 1 else [axes]

    for i in range(N):
        ax = axes[i]
        ax.plot(t, X_obs[:, i].numpy(), label="Obs", color="blue", alpha=0.6)
        ax.plot(t, X_int[:, i].numpy(), label="Int", color="red", alpha=0.6)

        # Highlight if this variable is intervened on
        if i in intervention.targets:
            ax.set_facecolor("#fff9e6")

        # Highlight intervention period
        if len(intervention.times) > 0:
            int_start = min(intervention.times)
            int_end = max(intervention.times)
            ax.axvspan(int_start, int_end, alpha=0.2, color="yellow")

        ax.set_title(f"Variable {i}" + (" (intervened)" if i in intervention.targets else ""))
        ax.set_xlabel("Time")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(N, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
