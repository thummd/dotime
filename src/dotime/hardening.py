"""Opt-in stability hardening for SCMs sampled by the generic prior.

Large generic SCMs diverge. Every edge weight is drawn from
``N(0, sigma_w^2)`` whatever a node's in-degree, so pre-activations grow with the
number of parents and compound along the instantaneous DAG, and the lagged
dynamics often have a reduced-form spectral radius above one (Appendix B of the
DoTime paper). At ``N_max=60, K_max=8`` most sampled episodes end up zeroed.

Three knobs address the three causes. The first two port the hardening that the
structured TSCM prior already uses for PFN training (``dotime.batched_tscm``) to
the per-node mechanism objects of the generic prior:

``unit_norm_rows``
    Rescale each variable's incoming weights, instantaneous and lagged together,
    to unit L2 norm, so a pre-activation no longer grows with the in-degree.
``spectral_rho``
    Scale every lagged weight by one common factor so that the spectral radius
    of the reduced-form companion matrix is at most ``spectral_rho``.
``bounded_square``
    Replace the ``x^2`` activation with ``tanh(x)^2``. Every other activation of
    the prior is bounded or 1-Lipschitz, but ``x^2`` is neither, so a lagged
    loop through it can blow up whatever the weights are.

The knobs rescale weights that were already sampled, or swap an activation that
was already chosen. They consume no random numbers, so a hardened prior draws
the same graphs, interventions and noise as an unhardened one with the same
seed. Only the weights a mechanism actually reads are touched, which keeps the
transform faithful to what ``TemporalMechanism.forward`` computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

HARDENING_KEYS = frozenset({"unit_norm_rows", "spectral_rho", "bounded_square"})

#: The combination validated at ``N_max=60, K_max=8`` (see
#: ``results/reference/hardening/``). Pass it as ``config["hardening"]``.
RECOMMENDED_HARDENING: dict[str, Any] = {
    "unit_norm_rows": True,
    "spectral_rho": 0.9,
    "bounded_square": True,
}

# Above this companion dimension a dense eigen-decomposition dominates the
# sampling cost (about 0.4 s at 480 x 480), so the largest-magnitude eigenvalue
# is taken from ARPACK instead.
_DENSE_EIG_MAX_DIM = 256
# The bisection stops once the achieved radius is within this relative
# distance below the cap, which keeps the lag scale close to the largest
# admissible one without spending evaluations on digits that do not matter.
_RHO_REL_TOL = 0.02
_MAX_BISECTIONS = 30


@dataclass(frozen=True)
class HardeningReport:
    """Outcome of hardening one linear system (one SCM, or one regime).

    Attributes
    ----------
    rho_before : float
        Reduced-form companion spectral radius after row normalization and
        before the lag rescaling.
    lag_scale : float
        Common factor applied to all lagged weights (1.0 when no rescaling was
        needed).
    rho_after : float
        Spectral radius after hardening.
    """

    rho_before: float
    lag_scale: float
    rho_after: float


def validate_hardening(config: Any) -> dict[str, Any] | None:
    """Check a ``hardening`` configuration and return it in canonical form.

    Parameters
    ----------
    config : dict or None
        The value of ``DoTime(config={"hardening": ...})``. ``None`` or an empty
        dict disables hardening.

    Returns
    -------
    dict or None
        ``{"unit_norm_rows": bool, "spectral_rho": float | None,
        "bounded_square": bool}``, or ``None`` when hardening is disabled.

    Raises
    ------
    TypeError
        If ``config`` is not a dict, or a value has the wrong type.
    ValueError
        If ``config`` has unknown keys or ``spectral_rho`` is not positive.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError(f"hardening must be a dict or None, got {type(config).__name__}")
    unknown = set(config) - HARDENING_KEYS
    if unknown:
        raise ValueError(
            f"unknown hardening keys {sorted(unknown)}; valid keys are {sorted(HARDENING_KEYS)}"
        )
    flags = {}
    for key in ("unit_norm_rows", "bounded_square"):
        value = config.get(key, False)
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a bool, got {type(value).__name__}")
        flags[key] = value
    rho = config.get("spectral_rho")
    if rho is not None:
        if isinstance(rho, bool) or not isinstance(rho, (int, float)):
            raise TypeError(f"spectral_rho must be a number or None, got {type(rho).__name__}")
        if not rho > 0:
            raise ValueError(f"spectral_rho must be positive, got {rho}")
        rho = float(rho)
    if not flags["unit_norm_rows"] and not flags["bounded_square"] and rho is None:
        return None
    return {
        "unit_norm_rows": flags["unit_norm_rows"],
        "spectral_rho": rho,
        "bounded_square": flags["bounded_square"],
    }


def _mechanisms(scm) -> list:
    """All mechanism objects of an SCM, across regimes for regime-switching SCMs.

    Parameters
    ----------
    scm : TemporalSCM or RegimeSwitchingTemporalSCM
        The sampled SCM.

    Returns
    -------
    list of TemporalMechanism
        Every mechanism, each listed once.
    """
    if hasattr(scm, "_regime_parents"):
        return [m for regime in scm.mechanisms for m in regime.values()]
    return list(scm.mechanisms.values())


def _systems(scm) -> list[tuple[list[list[tuple[torch.Tensor, int, int]]], int, int]]:
    """Collect the weights each mechanism reads, one linear system at a time.

    A plain ``TemporalSCM`` (diverse and chain SCMs) is one system. A
    regime-switching SCM contributes one system per regime.

    Parameters
    ----------
    scm : TemporalSCM or RegimeSwitchingTemporalSCM
        The sampled SCM.

    Returns
    -------
    list of tuple
        ``(rows, n, k_max)`` per system, where ``rows[i]`` lists
        ``(weight, parent_index, lag)`` for node ``i`` in topological order and
        ``lag == 0`` marks an instantaneous parent.
    """
    out = []
    if hasattr(scm, "_regime_parents"):
        for regime, parents in enumerate(scm._regime_parents):
            topo = parents["topo"]
            index = {v: i for i, v in enumerate(topo)}
            mechs = scm.mechanisms[regime]
            rows, k_max = [], 1
            for v in topo:
                m = mechs[v]
                # A parent without a weight under its name is not read by
                # forward(), so it is not part of the system being hardened.
                row = [
                    (m.weights_instant[p], index[p], 0)
                    for p in parents["instant"][v]
                    if p in m.weights_instant
                ]
                for k, lag_parents in enumerate(parents["lagged"][v]):
                    k_max = max(k_max, k + 1)
                    row += [
                        (m.weights_lagged[k][p], index[p], k + 1)
                        for p in lag_parents
                        if p in m.weights_lagged[k]
                    ]
                rows.append(row)
            out.append((rows, len(topo), k_max))
        return out

    rows, k_max = [], 1
    for i, v in enumerate(scm._topo):
        m = scm.mechanisms[v]
        row = [
            (m.weights_instant[p], j, 0)
            for p, j in scm._instant_parent_pairs[i]
            if p in m.weights_instant
        ]
        for k, pairs in enumerate(scm._lagged_parent_pairs[i]):
            k_max = max(k_max, k + 1)
            row += [
                (m.weights_lagged[k][p], j, k + 1) for p, j in pairs if p in m.weights_lagged[k]
            ]
        rows.append(row)
    out.append((rows, len(scm._topo), k_max))
    return out


def _lag_blocks(rows, n: int, k_max: int) -> list[np.ndarray]:
    """Reduced-form lag matrices ``A_k = (I - W_inst)^{-1} W_lag_k``.

    Parameters
    ----------
    rows : list
        Output rows of :func:`_systems` for one system.
    n : int
        Number of variables.
    k_max : int
        Number of lags.

    Returns
    -------
    list of numpy.ndarray
        One ``(n, n)`` matrix per lag.
    """
    w_inst = np.zeros((n, n))
    w_lag = [np.zeros((n, n)) for _ in range(k_max)]
    for i, row in enumerate(rows):
        for w, j, lag in row:
            if lag == 0:
                w_inst[i, j] = float(w.detach())
            else:
                w_lag[lag - 1][i, j] = float(w.detach())
    # W_inst is strictly lower triangular in topological order, so I - W_inst is
    # unit lower triangular and always invertible.
    inv = np.linalg.inv(np.eye(n) - w_inst)
    return [inv @ w for w in w_lag]


def _spectral_radius(blocks: list[np.ndarray], scale: float = 1.0) -> float:
    """Spectral radius of the block companion matrix of ``scale * A_k``.

    Parameters
    ----------
    blocks : list of numpy.ndarray
        Reduced-form lag matrices from :func:`_lag_blocks`.
    scale : float
        Common factor applied to every lag matrix.

    Returns
    -------
    float
        The largest eigenvalue magnitude.
    """
    n, k_max = blocks[0].shape[0], len(blocks)
    dim = n * k_max
    comp = np.zeros((dim, dim))
    comp[:n, :] = scale * np.hstack(blocks)
    if k_max > 1:
        comp[n:, : n * (k_max - 1)] = np.eye(n * (k_max - 1))
    if dim <= _DENSE_EIG_MAX_DIM:
        return float(np.max(np.abs(np.linalg.eigvals(comp))))
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigs

    try:
        # A fixed start vector keeps the result deterministic across runs.
        vals = eigs(comp, k=1, which="LM", v0=np.ones(dim), tol=1e-10, return_eigenvectors=False)
        return float(np.max(np.abs(vals)))
    except (ArpackNoConvergence, ArpackError):
        return float(np.max(np.abs(np.linalg.eigvals(comp))))


def companion_spectral_radius(scm) -> float:
    """Largest reduced-form companion spectral radius over an SCM's systems.

    This is the stability criterion of the convergence result: exact for linear
    mechanisms and an upper bound for the non-expansive activations of the prior.

    Parameters
    ----------
    scm : TemporalSCM or RegimeSwitchingTemporalSCM
        The sampled SCM.

    Returns
    -------
    float
        The spectral radius (0.0 if the SCM reads no lagged weights).
    """
    return max(
        (_spectral_radius(_lag_blocks(rows, n, k)) for rows, n, k in _systems(scm)),
        default=0.0,
    )


@torch.no_grad()
def harden_scm(
    scm,
    unit_norm_rows: bool = False,
    spectral_rho: float | None = None,
    bounded_square: bool = False,
) -> list[HardeningReport]:
    """Make an SCM's simulation stay bounded, in place.

    Parameters
    ----------
    scm : TemporalSCM or RegimeSwitchingTemporalSCM
        The sampled SCM, modified in place.
    unit_norm_rows : bool
        Normalize each variable's incoming weights to unit L2 norm first.
    spectral_rho : float, optional
        Cap on the reduced-form companion spectral radius, enforced by scaling
        all lagged weights of a system by one common factor.
    bounded_square : bool
        Replace every ``x^2`` activation with ``tanh(x)^2``.

    Returns
    -------
    list of HardeningReport
        One report per linear system (one per regime for regime-switching SCMs).
    """
    if bounded_square:
        # Imported here because prior imports this module at load time.
        from dotime._activations import TanhSquare
        from dotime.prior import Square

        for m in _mechanisms(scm):
            if isinstance(m.activation, Square):
                m.activation = TanhSquare()
    reports = []
    for rows, n, k_max in _systems(scm):
        if unit_norm_rows:
            for row in rows:
                norm = float(np.sqrt(sum(float(w.detach()) ** 2 for w, _, _ in row)))
                if norm > 0.0:
                    for w, _, _ in row:
                        w.div_(norm)
        blocks = _lag_blocks(rows, n, k_max)
        rho_before = _spectral_radius(blocks)
        scale, rho_after = 1.0, rho_before
        if spectral_rho is not None and rho_before > spectral_rho:
            # Invariant: rho(lo) <= cap < rho(hi). The result is admissible even
            # where the radius is not monotone in the scale.
            lo, hi, rho_lo = 0.0, 1.0, 0.0
            for _ in range(_MAX_BISECTIONS):
                mid = 0.5 * (lo + hi)
                rho_mid = _spectral_radius(blocks, mid)
                if rho_mid > spectral_rho:
                    hi = mid
                else:
                    lo, rho_lo = mid, rho_mid
                    if spectral_rho - rho_mid <= _RHO_REL_TOL * spectral_rho:
                        break
            scale, rho_after = lo, rho_lo
            for row in rows:
                for w, _, lag in row:
                    if lag > 0:
                        w.mul_(scale)
        reports.append(HardeningReport(rho_before, scale, rho_after))
    return reports
