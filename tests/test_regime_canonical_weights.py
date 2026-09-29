"""Tests for the opt-in regime-switching fix ``regime_canonical_weights``.

v1.0.0 regime-switching SCMs keep each mechanism's weights under the per-regime
node names (``x3``, ``u1``, ``y``) while passing parent values under canonical
names (``X0..X{N-1}``), so no parent is ever read. The fix re-keys the weights.
These tests pin that the default path is unchanged, that the flag draws the same
randomness, and that with it the regime dynamics come alive.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import torch.distributions as dist
from torch import nn

import dotime
from dotime import DoTime
from dotime._activations import Tanh
from dotime._sampling import TorchDistributionSampler
from dotime.interventions import InterventionSpec, InterventionType
from dotime.regime_switching import RegimeSwitchingTemporalSCM
from dotime.regime_switching_builder import RegimeSwitchingSCMBuilder
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.utils import DEFAULT_CONFIG

# Runs in a fresh interpreter against whichever ``dotime`` is first on
# PYTHONPATH and prints one SHA-256 per output. Only APIs that exist at the
# v1.0.0 release are used, so the same script fingerprints an older snapshot.
_FINGERPRINT_SCRIPT = r"""
import hashlib, json, sys, warnings
from pathlib import Path

import numpy as np
import torch

warnings.simplefilter("ignore", RuntimeWarning)
src = Path(sys.argv[1]).resolve()
import dotime

assert Path(dotime.__file__).resolve().is_relative_to(src), (dotime.__file__, str(src))
from dotime import DoTime
from dotime._build import episode_seed, make_episode
from dotime.regime_switching_builder import RegimeSwitchingSCMBuilder


def digest(*objs):
    h = hashlib.sha256()

    def feed(o):
        if isinstance(o, torch.Tensor):
            t = o.detach().cpu().contiguous()
            h.update(f"t{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.numpy().tobytes())
        elif isinstance(o, np.ndarray):
            h.update(f"a{o.dtype}{o.shape}".encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                feed(str(k))
                feed(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for x in o:
                feed(x)
            h.update(b"]")
        else:
            h.update(repr(o).encode())

    for o in objs:
        feed(o)
    return h.hexdigest()


def states(gen):
    np_state = np.random.get_state()
    return gen.get_state(), torch.get_rng_state(), np_state[1], np_state[2]


def episode(ep):
    return digest(ep.x_obs, ep.x_int, ep.y_true, ep.query_target, ep.intervention.to_dict())


out = {}
# The release path with the released seeds: one dot-RegimeSwitch-v1 episode per
# density, and dot-Generic-100k episodes 0-4, of which 2 and 4 are regime SCMs.
for d, tier, idx in ((2, 1, 0), (3, 2, 3333), (5, 3, 6666)):
    spec = {"kind": "regime", "idx": idx, "seed": episode_seed(20262719, idx), "T": 200,
            "num_regimes": d, "tier": tier}
    out[f"release_regime_{idx}"] = episode(make_episode(spec))
for idx in (0, 1, 2, 4):
    spec = {"kind": "generic", "idx": idx, "seed": episode_seed(20264719, idx), "T": 200}
    out[f"release_generic_{idx}"] = episode(make_episode(spec))
# The DoTime API, with the RNG states each call leaves behind.
for s in range(3):
    np.random.seed(s)
    torch.manual_seed(s)
    prior = DoTime(seed=s)
    xo, xi, iv, _ = prior.generate_regime_pair(T=60, num_regimes=3)
    out[f"regime_pair_{s}"] = digest(xo, xi, iv.to_dict(), states(prior.generator))
for s in (0, 3, 15, 16, 31):  # diverse, chain, then three regime-switching SCMs
    np.random.seed(s)
    torch.manual_seed(s)
    prior = DoTime(config={"N_max": 6}, seed=s)
    xo, xi, iv, scm = prior.generate_pair(T=40)
    out[f"pair_{s}"] = digest(type(scm).__name__, xo, xi, iv.to_dict(), states(prior.generator))
# The builder directly, including the regime path.
np.random.seed(7)
torch.manual_seed(7)
g = torch.Generator()
g.manual_seed(7)
builder = RegimeSwitchingSCMBuilder(num_nodes=5, max_lag=2, activations=DoTime(seed=7).activations,
                                    gamma=0.7, sigma_w=1.0, sigma_b=0.5)
scm = builder.sample(g, num_regimes=2)
xo, regimes = scm.sample_observational(T=60, burn_in=20, generator=g, return_regimes=True)
out["builder"] = digest(xo, regimes, scm.transition_matrix, states(g))
print(json.dumps(out))
"""


def _prior(seed: int, flag: bool, **overrides) -> DoTime:
    """Build a prior the way the release build does, with the global numpy RNG seeded too.

    Parameters
    ----------
    seed : int
        Seed for the global torch RNG, the global numpy RNG and the prior.
    flag : bool
        Value of ``config["regime_canonical_weights"]``.
    **overrides
        Further config entries.

    Returns
    -------
    DoTime
        The seeded prior.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = {**DEFAULT_CONFIG, **overrides, "regime_canonical_weights": flag}
    return DoTime(config=config, seed=seed)


def _builder(flag: bool, num_nodes: int = 5, max_lag: int = 2) -> RegimeSwitchingSCMBuilder:
    """A regime builder with the prior's default activations and scales.

    Parameters
    ----------
    flag : bool
        Value of ``canonical_weights``.
    num_nodes : int
        Number of variables.
    max_lag : int
        Maximum lag.

    Returns
    -------
    RegimeSwitchingSCMBuilder
        The builder.
    """
    return RegimeSwitchingSCMBuilder(
        num_nodes=num_nodes,
        max_lag=max_lag,
        activations=DoTime(seed=0).activations,
        gamma=DEFAULT_CONFIG["gamma"],
        sigma_w=DEFAULT_CONFIG["sigma_w"],
        sigma_b=DEFAULT_CONFIG["sigma_b"],
        canonical_weights=flag,
    )


def _sample(flag: bool, seed: int, num_regimes: int = 2) -> RegimeSwitchingTemporalSCM:
    """Sample a small regime SCM with fixed seeds.

    Parameters
    ----------
    flag : bool
        Value of ``canonical_weights``.
    seed : int
        Seed for the global torch RNG and the sampling generator.
    num_regimes : int
        Number of regimes.

    Returns
    -------
    RegimeSwitchingTemporalSCM
        The sampled SCM.
    """
    torch.manual_seed(seed)
    return _builder(flag).sample(torch.Generator().manual_seed(seed), num_regimes=num_regimes)


def _abs_acf1(x: torch.Tensor) -> torch.Tensor:
    """Per-variable ``|corr(x[t], x[t-1])|`` of a ``(T, N)`` trajectory.

    Parameters
    ----------
    x : torch.Tensor
        Trajectory.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``.
    """
    x = x - x.mean(0)
    return ((x[1:] * x[:-1]).sum(0) / (x * x).sum(0)).abs()


def _other_children(scm: RegimeSwitchingTemporalSCM, index: int) -> int:
    """Count the edges from variable ``index`` into other variables, over all regimes and lags.

    Parameters
    ----------
    scm : RegimeSwitchingTemporalSCM
        The SCM.
    index : int
        Position of the variable in the canonical order.

    Returns
    -------
    int
        Number of outgoing edges to other variables.
    """
    total = 0
    for dag in scm.dags:
        total += dag.G_0.out_degree(dag.topo_order[index])
        for g_k in dag.G_lags:
            total += int((g_k[index] > 0).sum()) - int(g_k[index, index] > 0)
    return total


def test_rename_nodes_is_an_exact_relabelling():
    """Re-keying keeps the Parameters and their order, so the output is bit-identical."""
    mech = TemporalMechanism(
        ["y", "u0", "x3"],
        Tanh(),
        num_lags=2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )
    inst = {"u0": torch.tensor([0.7]), "x3": torch.tensor([-1.2])}
    lagged = [{"y": torch.tensor([0.4]), "x3": torch.tensor([2.0])}, {"u0": torch.tensor([-0.3])}]
    eps = torch.tensor([0.05])
    before = mech(inst, lagged, eps)
    params = {id(p) for p in mech.parameters()}
    keys = [list(d) for d in (mech.weights_instant, *mech.weights_lagged)]
    rng = torch.get_rng_state()

    mapping = {"y": "X0", "u0": "X1", "x3": "X2"}
    mech.rename_nodes(mapping)

    def renamed(d):
        return {mapping[k]: v for k, v in d.items()}

    assert torch.equal(torch.get_rng_state(), rng)
    assert {id(p) for p in mech.parameters()} == params
    assert [list(d) for d in (mech.weights_instant, *mech.weights_lagged)] == [
        [mapping[k] for k in ks] for ks in keys
    ]
    assert torch.equal(mech(renamed(inst), [renamed(d) for d in lagged], eps), before)
    # The v1.0.0 regime failure: parents under names the weights do not carry
    # are skipped, and the mechanism returns only its noise term.
    assert torch.equal(mech(inst, lagged, eps), eps)


def test_rename_nodes_rejects_bad_mappings_without_side_effects():
    mech = TemporalMechanism(["a", "b"], Tanh(), num_lags=1, device=torch.device("cpu"))
    with pytest.raises(KeyError, match="no entry"):
        mech.rename_nodes({"a": "X0"})
    with pytest.raises(ValueError, match="one name"):
        mech.rename_nodes({"a": "X0", "b": "X0"})
    assert list(mech.weights_instant) == ["a", "b"]
    assert list(mech.weights_lagged[0]) == ["a", "b"]


def test_regime_mechanisms_read_their_parents_only_with_the_flag():
    """Called the way the SCM calls them, v1.0.0 mechanisms return exactly ``eps``."""
    eps = torch.tensor([0.25])
    for flag in (False, True):
        scm = _sample(flag, seed=0, num_regimes=3)
        outputs = []
        for regime, mechs in enumerate(scm.mechanisms):
            parents = scm._regime_parents[regime]
            for v in parents["topo"]:
                inst = {p: torch.tensor(1.5) for p in parents["instant"][v]}
                lagged = [{p: torch.tensor(-0.5) for p in ps} for ps in parents["lagged"][v]]
                if inst or any(lagged):
                    outputs.append(mechs[v](inst, lagged, eps))
        assert len(outputs) >= 5
        noise_only = sum(torch.equal(o, eps) for o in outputs)
        if flag:
            assert noise_only < len(outputs) / 2
        else:
            assert noise_only == len(outputs)


def test_default_path_is_bitwise_identical_to_committed_head(tmp_path):
    """With the flag off, the prior reproduces ``git archive HEAD src`` bit for bit.

    Both trees run one fingerprint script in fresh interpreters: the release
    ``make_episode`` path with the released suite seeds, the ``DoTime`` API with
    the RNG states it leaves behind, and the regime builder. Once a change is
    committed this compares HEAD with itself, so it guards uncommitted edits.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    repo = Path(__file__).resolve().parents[1]
    try:
        tar = subprocess.run(
            [git, "-C", str(repo), "archive", "--format=tar", "HEAD", "src"],
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"cannot snapshot HEAD: {exc.stderr.decode().strip()}")
    head = tmp_path / "head"
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(head, filter="data")
        else:  # pragma: no cover - interpreters without PEP 706 extraction filters
            archive.extractall(head)
    script = tmp_path / "fingerprint.py"
    script.write_text(_FINGERPRINT_SCRIPT)
    current = Path(dotime.__file__).resolve().parents[1]

    procs = []
    for src in (head / "src", current):
        path = os.pathsep.join(p for p in (str(src), os.environ.get("PYTHONPATH", "")) if p)
        env = {**os.environ, "PYTHONPATH": path, "OMP_NUM_THREADS": "1"}
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script), str(src)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
        )
    fingerprints = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=900)
        assert proc.returncode == 0, stderr
        fingerprints.append(json.loads(stdout))
    at_head, now = fingerprints
    assert at_head.keys() == now.keys()
    changed = sorted(k for k in at_head if at_head[k] != now[k])
    assert not changed, f"the default path no longer reproduces HEAD for {changed}"


@pytest.mark.parametrize("path", ["regime_pair", "pair"])
def test_flag_draws_the_same_randomness(path):
    """The flag re-keys sampled weights only: same draws, same RNG states afterwards."""
    kinds = set()
    # generate_pair: seed 0 draws a diverse SCM, 3 and 11 chains, 15 to 35 regime SCMs.
    for seed in (0, 3, 11, 15, 16, 31, 35) if path == "pair" else range(4):
        runs = []
        for flag in (False, True):
            prior = _prior(seed, flag, N_max=6)
            if path == "regime_pair":
                out = prior.generate_regime_pair(T=40, num_regimes=2 + seed % 2)
            else:
                out = prior.generate_pair(T=40)
            runs.append(
                (out, prior.generator.get_state(), torch.get_rng_state(), np.random.get_state())
            )
        (xo0, xi0, iv0, scm0), gen0, glob0, np0 = runs[0]
        (xo1, xi1, iv1, scm1), gen1, glob1, np1 = runs[1]
        # to_dict() encodes scalar, tensor and time-varying profile values alike.
        assert iv0.to_dict() == iv1.to_dict()
        assert torch.equal(gen0, gen1)
        assert torch.equal(glob0, glob1)
        assert np.array_equal(np0[1], np1[1])
        assert np0[2:] == np1[2:]
        assert type(scm0) is type(scm1)
        kinds.add(type(scm0).__name__)
        if not isinstance(scm0, RegimeSwitchingTemporalSCM):
            # The flag touches regime-switching SCMs only.
            assert torch.equal(xo0, xo1)
            assert torch.equal(xi0, xi1)
            continue
        np.testing.assert_array_equal(scm0.transition_matrix, scm1.transition_matrix)
        for dag0, dag1, mechs0, mechs1 in zip(
            scm0.dags, scm1.dags, scm0.mechanisms, scm1.mechanisms, strict=True
        ):
            assert set(dag0.G_0.edges) == set(dag1.G_0.edges)
            assert all(np.array_equal(a, b) for a, b in zip(dag0.G_lags, dag1.G_lags, strict=True))
            for v in dag1.topo_order:
                m0, m1 = mechs0[v], mechs1[v]
                assert set(m0.weights_instant).isdisjoint(dag1.topo_order)
                assert set(m1.weights_instant) == set(dag1.topo_order)
                for w0, w1 in zip(
                    (m0.weights_instant, *m0.weights_lagged),
                    (m1.weights_instant, *m1.weights_lagged),
                    strict=True,
                ):
                    values0 = torch.cat([w.detach() for w in w0.values()])
                    values1 = torch.cat([w.detach() for w in w1.values()])
                    assert torch.equal(values0, values1)
                assert torch.equal(m0.bias, m1.bias)
    if path == "pair":
        assert {"RegimeSwitchingTemporalSCM", "TemporalSCM"} <= kinds


def test_flag_gives_regime_trajectories_lag_structure():
    """v1.0.0 regime trajectories are white noise. With the flag they have memory."""
    acf = {False: [], True: []}
    for seed in range(8):
        for flag in (False, True):
            x_obs, _, _, _ = _prior(seed, flag, N_max=5).generate_regime_pair(T=200, num_regimes=2)
            if float(x_obs.abs().max()) > 0:
                acf[flag].append(_abs_acf1(x_obs))
    assert len(acf[False]) == 8
    assert len(acf[True]) >= 3
    off, on = torch.cat(acf[False]), torch.cat(acf[True])
    # White noise at T=200 has sd(acf1) of about 0.07, so 0.3 is over 4 sd.
    assert float(off.max()) < 0.3
    assert float((on > 0.3).float().mean()) > 0.25


def test_flag_propagates_interventions_under_shared_noise():
    """An intervention moves other variables with the flag, and never without it.

    A soft intervention still draws the treated variable's noise, so reseeding
    the generator and the global numpy RNG before each arm gives both arms the
    same noise and the same regime path.
    """
    onset, t_len, burn_in = 20, 80, 30
    moved_with_flag = 0
    for seed in range(6):
        for flag in (False, True):
            scm = _sample(flag, seed)
            n = len(scm.dags[0].topo_order)
            target = max(range(n), key=lambda i: _other_children(scm, i))
            assert _other_children(scm, target) > 0
            iv = InterventionSpec(
                targets=[target],
                times=list(range(onset, 60)),
                intervention_type=InterventionType.SOFT,
                values=3.0,
            )
            arms = []
            for intervened in (False, True):
                np.random.seed(11)
                g = torch.Generator().manual_seed(12)
                if intervened:
                    arms.append(scm.sample_interventional(t_len, iv, burn_in=burn_in, generator=g))
                else:
                    arms.append(scm.sample_observational(t_len, burn_in=burn_in, generator=g))
            x_obs, x_int = arms
            if float(x_obs.abs().max()) == 0 or float(x_int.abs().max()) == 0:
                continue  # diverged; zeroed arms carry no dynamics to compare
            assert torch.equal(x_obs[:onset], x_int[:onset])
            others = [j for j in range(n) if j != target]
            moved = float((x_int[onset:, others] - x_obs[onset:, others]).abs().max())
            if flag:
                assert moved > 1e-3
                moved_with_flag += 1
            else:
                assert moved == 0.0
    assert moved_with_flag >= 3


def _exploding_scm(threshold: float | None) -> RegimeSwitchingTemporalSCM:
    """One variable with ``x_t = 2 x_{t-1} + 1 + eps``, which hits the clip within ~10 steps.

    Parameters
    ----------
    threshold : float or None
        The SCM's ``divergence_threshold``.

    Returns
    -------
    RegimeSwitchingTemporalSCM
        A single-regime SCM.
    """
    mech = TemporalMechanism(["X0"], nn.Identity(), num_lags=1, device=torch.device("cpu"))
    with torch.no_grad():
        mech.weights_lagged[0]["X0"].fill_(2.0)
        mech.bias.fill_(1.0)
    g_0 = nx.DiGraph()
    g_0.add_node("X0")
    dag = TemporalDAG(G_0=g_0, G_lags=[np.ones((1, 1), dtype=np.float32)], K=1, topo_order=["X0"])
    noise = {"X0": TorchDistributionSampler(dist.Normal(0.0, 0.1))}
    return RegimeSwitchingTemporalSCM(
        [dag], [{"X0": mech}], noise, np.ones((1, 1)), divergence_threshold=threshold
    )


def test_divergence_threshold_zeroes_the_arm_without_changing_the_draws():
    """v1.0.0 lets an exploding arm saturate at the clip. The threshold zeroes it."""
    results = []
    for threshold in (None, 500.0):
        scm = _exploding_scm(threshold)
        np.random.seed(3)
        g = torch.Generator().manual_seed(4)
        if threshold is None:
            x = scm.sample_observational(100, burn_in=10, generator=g)
        else:
            with pytest.warns(RuntimeWarning, match="Regime-switching SCM diverged"):
                x = scm.sample_observational(100, burn_in=10, generator=g)
        results.append((x, g.get_state(), np.random.get_state()[1]))
    (x_legacy, gen0, np0), (x_flagged, gen1, np1) = results
    assert float(x_legacy.max()) == 1000.0
    assert torch.equal(x_flagged, torch.zeros(100, 1))
    # The check runs after the simulation, so a diverged arm consumes exactly
    # the draws of a finished one and later episodes keep their randomness.
    assert torch.equal(gen0, gen1)
    assert np.array_equal(np0, np1)

    scm = _exploding_scm(500.0)
    iv = InterventionSpec(
        targets=[0], times=[5], intervention_type=InterventionType.HARD, values=0.0
    )
    with pytest.warns(RuntimeWarning, match="Regime-switching SCM diverged"):
        x_int, regimes = scm.sample_interventional(
            100, iv, burn_in=10, generator=torch.Generator().manual_seed(4), return_regimes=True
        )
    assert torch.equal(x_int, torch.zeros(100, 1))
    assert regimes.shape == (100,)
    with pytest.raises(ValueError, match="positive"):
        _exploding_scm(0.0)


def test_regime_share_is_decided_by_the_first_draw():
    """The datasheet identifies Generic-100k's regime share from each episode's seed.

    ``DoTime.sample_scm`` picks a regime-switching SCM exactly when the first
    draw of the freshly seeded generator lies in ``[0.15, 0.30)``.
    """
    kinds = []
    for seed in range(40):
        first = torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
        is_regime = isinstance(DoTime(seed=seed).sample_scm(), RegimeSwitchingTemporalSCM)
        assert is_regime == (0.15 <= first < 0.30), seed
        kinds.append(is_regime)
    assert any(kinds)
    assert not all(kinds)


def test_builder_enables_the_divergence_check_only_with_canonical_weights():
    assert _sample(False, seed=0).divergence_threshold is None
    assert _sample(True, seed=0).divergence_threshold == 500.0


def test_regime_canonical_weights_must_be_a_bool():
    assert DoTime(seed=0).regime_canonical_weights is False
    assert DoTime(config={"regime_canonical_weights": True}, seed=0).regime_canonical_weights
    for bad in ("false", 1, None):
        with pytest.raises(TypeError, match="regime_canonical_weights"):
            DoTime(config={"regime_canonical_weights": bad}, seed=0)
