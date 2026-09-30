"""Tests for ground-truth graph metadata (``dotime.graph_meta`` and ``record_graph``).

The lag convention is pinned by simulation rather than by re-reading the
builder, the extractors are checked against the structures the generators
actually release, and recording a graph must leave every tensor and random
stream of an episode exactly as it was.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import torch.distributions as dist
from torch import nn

from dotime import DoTime
from dotime._build import _forward_opt_in, episode_specs, make_episode
from dotime._sampling import TorchDistributionSampler
from dotime.continuous.continuous_scm import ContinuousIntervention, InterventionKind
from dotime.continuous.tscm_sampler import ContinuousTSCMSampler
from dotime.extended import TSCMPrior
from dotime.graph_meta import LaggedGraph, PathLag, load_graph_sidecar, path_lag
from dotime.interventions import InterventionSpec, InterventionType
from dotime.regime_switching import RegimeSwitchingTemporalSCM
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.temporal_scm import TemporalSCM
from dotime.tscm_sampler import TSCMSampler, TSCMStructure

# DoTime(config={"N_max": 6}, seed=s).sample_scm() draws a diverse, a chain and
# a regime-switching SCM for these seeds (the class is asserted, so a change to
# the prior fails loudly instead of silently testing another family).
_FAMILY_SEEDS = {"diverse": 0, "chain": 3, "regime": 16}

_REPO = Path(__file__).resolve().parents[1]
_SIDECARS = {
    "dot-Generic-100k": "dot-Generic-100k-v1.0.0_graph.jsonl.gz",
    "dot-RegimeSwitch-v1": "dot-RegimeSwitch-v1.0.0_graph.jsonl.gz",
}


def _prior_scm(family: str, **config):
    """Sample the SCM that ``_FAMILY_SEEDS`` assigns to a family.

    The global torch and numpy RNGs are seeded too, as ``make_episode`` does,
    because the builders draw edge probabilities from the global torch RNG.

    Args:
        family: ``"diverse"``, ``"chain"`` or ``"regime"``.
        **config: Extra DoTime config entries.

    Returns:
        The sampled SCM.
    """
    seed = _FAMILY_SEEDS[family]
    torch.manual_seed(seed)
    np.random.seed(seed)
    scm = DoTime(config={"N_max": 6, **config}, seed=seed).sample_scm()
    expected = RegimeSwitchingTemporalSCM if family == "regime" else TemporalSCM
    assert type(scm) is expected
    return scm


def _structural_edges(topo, g0, g_lags) -> set[tuple[int, int, int]]:
    """Edges of a temporal DAG in topological coordinates, read independently.

    Args:
        topo: Node names in topological order.
        g0: Instantaneous networkx graph.
        g_lags: Lag adjacency matrices, ``g_lags[k][j, i] > 0`` for
            ``topo[j](t - k - 1) -> topo[i](t)``.

    Returns:
        ``(src, dst, lag)`` triples.
    """
    idx = {v: i for i, v in enumerate(topo)}
    edges = {(idx[u], idx[v], 0) for u, v in g0.edges()}
    for k, g_k in enumerate(g_lags):
        rows, cols = np.nonzero(np.asarray(g_k) > 0)
        edges |= {(int(j), int(i), k + 1) for j, i in zip(rows, cols, strict=True)}
    return edges


def _rng_states(gen: torch.Generator):
    """Snapshot the global torch, global numpy and one generator's state.

    Args:
        gen: A torch generator.

    Returns:
        Tuple of copies that later calls cannot mutate.
    """
    np_state = np.random.get_state()
    return torch.get_rng_state().clone(), gen.get_state().clone(), np_state[1].copy(), np_state[2]


# --------------------------------------------------------------------------- #
# Lag convention
# --------------------------------------------------------------------------- #


def test_lag_two_edge_delays_the_response_by_exactly_two_steps():
    nodes = ["X0", "X1"]
    g0 = nx.DiGraph()
    g0.add_nodes_from(nodes)
    lag2 = np.zeros((2, 2), dtype=np.float32)
    lag2[0, 1] = 1.0  # X0(t-2) -> X1(t), the only edge
    dag = TemporalDAG(G_0=g0, G_lags=[np.zeros((2, 2), np.float32), lag2], K=2, topo_order=nodes)
    gen = torch.Generator().manual_seed(0)
    mechs = {
        v: TemporalMechanism(nodes, nn.Identity(), 2, torch.device("cpu"), generator=gen)
        for v in nodes
    }
    with torch.no_grad():
        mechs["X1"].weights_lagged[1]["X0"].fill_(1.0)
    noise = {v: TorchDistributionSampler(dist.Normal(0.0, 0.1)) for v in nodes}
    scm = TemporalSCM(dag, mechs, noise)

    # Shared noise, so the arms differ only through the one-step intervention.
    t_len, burn_in, onset = 30, 10, 12
    scm.freeze_noise(t_len + burn_in, generator=torch.Generator().manual_seed(1))
    x_obs = scm.sample_observational(T=t_len, burn_in=burn_in)
    iv = InterventionSpec([0], [onset], InterventionType.HARD, 5.0)
    x_int = scm.sample_interventional(T=t_len, intervention=iv, burn_in=burn_in)
    changed = (x_int - x_obs).abs() > 0
    assert changed[:, 0].nonzero().flatten().tolist() == [onset]
    assert changed[:, 1].nonzero().flatten().tolist() == [onset + 2]

    graph = LaggedGraph.from_scm(scm)
    assert graph.edges == ((0, 1, 2),)
    assert (graph.k_sampled, graph.k_eff) == (2, 2)
    assert path_lag(graph, [0], 1) == PathLag(True, 2, 1, (2,))


def test_continuous_edges_respond_one_observation_step_later():
    sampler = ContinuousTSCMSampler(TSCMStructure("mediator"))
    scm = sampler.sample(generator=torch.Generator().manual_seed(0))
    times = torch.arange(40, dtype=torch.float32)
    onset = 10
    iv = ContinuousIntervention(
        target=sampler.get_intervention_target(),
        t_start=float(onset),
        t_end=40.0,
        kind=InterventionKind.HARD,
        value=3.0,
    )
    _, x_obs, x_cf = scm.sample_counterfactual_pair(
        times, times.diff(), iv, generator=torch.Generator().manual_seed(1)
    )
    changed = (x_cf - x_obs).abs() > 0
    graph = LaggedGraph.from_continuous_structure("mediator")
    first_change = {
        name: int(changed[:, sampler.node_names.index(name)].nonzero()[0]) - onset
        for name in graph.columns
    }
    # A is clamped at the onset, M responds one step later and Y one after M.
    assert first_change == {"A": 0, "M": 1, "Y": 2}
    assert path_lag(graph, [0], graph.columns.index("Y")).min_lag == 2
    assert {lag for _, _, lag in graph.edges} == {1}


# --------------------------------------------------------------------------- #
# Extraction from the generic prior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ["diverse", "chain"])
def test_temporal_scm_graph_is_the_sampled_graph(family):
    scm = _prior_scm(family)
    graph = LaggedGraph.from_scm(scm)
    expected = _structural_edges(scm._topo, scm._G_0, scm._G_lags)
    assert set(graph.edges) == expected
    assert graph.columns == tuple(scm._topo)
    assert graph.reads_parents is True
    assert graph.regime_edges is None
    assert graph.k_sampled == scm._K
    assert graph.k_eff == max(lag for _, _, lag in expected)
    assert graph.hidden == graph.latent == ()
    if family == "chain":
        assert graph.k_sampled == 2
        assert {(s, d) for s, d, lag in graph.edges if lag == 0} == {
            (i, i + 1) for i in range(graph.n - 1)
        }
        assert all(s == d for s, d, lag in graph.edges if lag > 0)
    else:
        assert "y" in graph.columns


def test_default_regime_scm_reads_no_parent():
    scm = _prior_scm("regime")
    graph = LaggedGraph.from_scm(scm)
    assert graph.reads_parents is False
    assert graph.edges == ()
    assert graph.k_eff == 0
    assert graph.columns == tuple(f"X{i}" for i in range(graph.n))
    assert graph.regime_edges is not None
    assert len(graph.regime_edges) == scm.num_regimes
    for regime_edges, dag in zip(graph.regime_edges, scm.dags, strict=True):
        assert set(regime_edges) == _structural_edges(dag.topo_order, dag.G_0, dag.G_lags)
    assert any(graph.regime_edges)


def test_canonical_weights_make_the_same_regime_graphs_effective():
    default = LaggedGraph.from_scm(_prior_scm("regime"))
    fixed = LaggedGraph.from_scm(_prior_scm("regime", regime_canonical_weights=True))
    assert fixed.reads_parents is True
    # The flag draws the same random numbers, so the sampled graphs agree.
    assert fixed.regime_edges == default.regime_edges
    assert set(fixed.edges) == set().union(*map(set, fixed.regime_edges))
    assert fixed.k_eff == max(lag for _, _, lag in fixed.edges)


def test_released_columns_can_leave_nodes_latent():
    scm = _prior_scm("chain")
    names = list(scm._topo)
    released = [names[0], names[-1]]
    graph = LaggedGraph.from_scm(scm, columns=released, hidden=[1])
    assert graph.n == 2
    assert graph.latent == tuple(names[1:-1])
    assert graph.hidden == (1,)
    # The chain's first node reaches its last only through latent nodes.
    assert not any(s < 2 and d < 2 and s != d for s, d, _ in graph.edges)
    assert path_lag(graph, [0], 1) == PathLag(True, 0, len(names) - 1, ())
    with pytest.raises(ValueError, match="distinct nodes"):
        LaggedGraph.from_scm(scm, columns=[names[0], names[0]])
    with pytest.raises(ValueError, match="distinct nodes"):
        LaggedGraph.from_scm(scm, columns=["not_a_node"])
    with pytest.raises(TypeError, match="cannot extract"):
        LaggedGraph.from_scm(object())


# --------------------------------------------------------------------------- #
# Named structures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("structure", list(TSCMStructure))
def test_named_structure_columns_follow_the_release(structure):
    graph = LaggedGraph.from_structure(structure.value)
    topo = TSCMSampler(structure, max_lag=1)._build_dag().topo_order
    assert graph.columns == tuple(topo[i] for i in TSCMPrior(structure).canonical_perm)
    assert graph.columns[0] == "A"
    assert graph.columns[-1] == "Y"
    assert graph.hidden == ((graph.columns.index("U"),) if "U" in graph.columns else ())
    assert graph.reads_parents is True
    assert (graph.k_sampled, graph.k_eff, graph.time) == (1, 1, "discrete")
    # A sampled SCM of the structure reads exactly these edges.
    scm = TSCMSampler(structure, max_lag=1).sample(torch.Generator().manual_seed(0))
    sampled = LaggedGraph.from_scm(scm, columns=graph.columns, hidden=graph.hidden)
    assert sampled.edges == graph.edges

    continuous = LaggedGraph.from_continuous_structure(structure.value)
    assert continuous.columns == graph.columns
    assert continuous.hidden == graph.hidden
    assert continuous.time == "continuous"
    assert {lag for _, _, lag in continuous.edges} == {1}
    # Collapsing G_0 and lag 1 keeps every cross-variable dependence.
    assert {(s, d) for s, d, _ in continuous.edges if s != d} == {
        (s, d) for s, d, _ in graph.edges if s != d
    }


@pytest.mark.parametrize("structure", list(TSCMStructure))
@pytest.mark.parametrize("kind", ["identifiability", "continuous"])
def test_recorded_structure_graph_matches_the_released_episode(structure, kind):
    spec = {"kind": kind, "idx": 0, "seed": 5, "T": 40, "structure": structure.value, "tier": 1}
    ep = make_episode({**spec, "record_graph": True})
    graph = LaggedGraph.from_dict(ep.metadata["graph"])
    assert graph.n == ep.n_vars
    assert ep.intervention.targets == [graph.columns.index("A")] == [0]
    if kind == "identifiability":
        assert ep.query_target.tolist() == [graph.columns.index("Y")]
    else:  # continuous queries any observable variable, the treatment included
        assert not set(ep.query_target.tolist()) & set(graph.hidden)
    for h in graph.hidden:
        assert graph.columns[h] == "U"
        assert float(ep.x_obs[:, h].abs().max()) == 0.0


def test_legacy_structure_label_is_accepted():
    assert LaggedGraph.from_structure("rct_no_confounding") == LaggedGraph.from_structure(
        "bi_variate"
    )


# --------------------------------------------------------------------------- #
# path_lag
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        ("mediator", PathLag(True, 1, 2, ())),
        ("back_door", PathLag(True, 0, 1, (0,))),
        ("front_door", PathLag(True, 0, 2, ())),
        ("unobserved_confounder", PathLag(False, None, None, ())),
        ("observed_confounder", PathLag(False, None, None, ())),
    ],
)
def test_path_lag_on_named_structures(structure, expected):
    graph = LaggedGraph.from_structure(structure)
    assert path_lag(graph, [0], graph.n - 1) == expected


def test_path_lag_separates_fewest_hops_from_smallest_lag():
    graph = LaggedGraph(
        n=5,
        columns=("a", "b", "c", "d", "e"),
        edges=((0, 3, 5), (0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 3, 2), (4, 0, 1)),
        k_sampled=5,
        k_eff=5,
    )
    assert path_lag(graph, [0], 3) == PathLag(True, 1, 1, (5,))
    # Several sources: the best path over all of them, direct lags pooled.
    assert path_lag(graph, [1, 4], 3) == PathLag(True, 1, 2, ())
    # A queried source is reached at once; self-edges never make a path.
    assert path_lag(graph, [3], 3) == PathLag(True, 0, 0, ())
    assert path_lag(graph, [3], 0) == PathLag(False, None, None, ())
    with pytest.raises(ValueError, match="at least one source"):
        path_lag(graph, [], 3)
    with pytest.raises(ValueError, match="outside the graph"):
        path_lag(graph, [0], 5)


# --------------------------------------------------------------------------- #
# Randomness and serialization
# --------------------------------------------------------------------------- #


def test_extraction_leaves_every_rng_untouched():
    scms = [_prior_scm(family) for family in _FAMILY_SEEDS]
    torch.manual_seed(123)
    np.random.seed(123)
    gen = torch.Generator().manual_seed(7)
    before = _rng_states(gen)
    for scm in scms:
        graph = LaggedGraph.from_scm(scm)
        path_lag(graph, [0], graph.n - 1)
        LaggedGraph.from_dict(graph.to_dict())
    for structure in TSCMStructure:
        LaggedGraph.from_structure(structure.value)
        LaggedGraph.from_continuous_structure(structure.value)
    after = _rng_states(gen)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])
    assert before[3] == after[3]


_KIND_CONFIGS = {
    "generic": ({"generator": "generic", "n_episodes": 3, "T": 60}, 11),
    "regime": ({"generator": "regime", "densities": {2: 1, 3: 2}, "n_episodes": 2, "T": 60}, 12),
    "identifiability": (
        {
            "generator": "identifiability",
            "episodes_per_structure": 1,
            "T": 60,
            "structures": {"back_door": 1, "mediator": 2, "front_door": 2},
        },
        13,
    ),
    "continuous": (
        {"generator": "continuous", "n_episodes": 2, "T": 60, "structures": ["front_door"]},
        14,
    ),
}


@pytest.mark.parametrize("kind", list(_KIND_CONFIGS))
def test_record_graph_changes_nothing_but_the_graph_entry(kind):
    cfg, seed = _KIND_CONFIGS[kind]
    plain_specs = episode_specs(cfg, seed, 1.0)
    graph_specs = episode_specs({**cfg, "record_graph": True}, seed, 1.0)
    assert all("record_graph" not in s for s in plain_specs)
    assert all(
        g == {**p, "record_graph": True} for p, g in zip(plain_specs, graph_specs, strict=True)
    )
    for plain_spec, graph_spec in zip(plain_specs, graph_specs, strict=True):
        plain, recorded = make_episode(plain_spec), make_episode(graph_spec)
        for field in ("x_obs", "x_int", "y_true", "query_target", "query_time"):
            assert torch.equal(getattr(plain, field), getattr(recorded, field)), field
        assert plain.intervention.to_dict() == recorded.intervention.to_dict()
        meta = dict(recorded.metadata)
        stored = meta.pop("graph")
        assert "graph" not in plain.metadata
        assert meta.keys() == plain.metadata.keys()
        for key, value in plain.metadata.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, meta[key]), key
            else:
                assert value == meta[key], key
        graph = LaggedGraph.from_dict(stored)
        assert graph.n == recorded.n_vars
        sources = recorded.intervention.targets
        assert stored["path"] == [
            {"target": q, **path_lag(graph, sources, q).to_dict()}
            for q in recorded.query_target.tolist()
        ]
        json.dumps(stored)  # built-in values only


def test_record_graph_flag_must_be_a_bool():
    spec = episode_specs({"generator": "generic", "n_episodes": 1, "T": 60}, 11, 1.0)[0]
    with pytest.raises(TypeError, match="record_graph"):
        make_episode({**spec, "record_graph": "false"})


def test_opt_in_keys_are_forwarded_only_when_set():
    specs = [{"idx": 0}, {"idx": 1}]
    assert _forward_opt_in({"generator": "generic"}, specs) is specs
    assert _forward_opt_in({"record_graph": True}, specs) == [
        {"idx": 0, "record_graph": True},
        {"idx": 1, "record_graph": True},
    ]


def test_graph_metadata_survives_write_and_read_suite(tmp_path):
    pytest.importorskip("pyarrow")
    from dotime import _release_io
    from dotime.benchmarks import SuiteMetadata

    episodes = []
    for kind in ("generic", "identifiability"):
        cfg, seed = _KIND_CONFIGS[kind]
        episodes += [
            make_episode(s) for s in episode_specs({**cfg, "record_graph": True}, seed, 1.0)
        ]
    meta = SuiteMetadata(
        name="graph-test",
        version="0.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="",
        n_episodes=len(episodes),
    )
    _release_io.write_suite(meta, episodes, tmp_path / "suite", package_version="test", seed=0)
    suite = _release_io.read_suite(meta, tmp_path / "suite")
    for written, read in zip(episodes, suite, strict=True):
        assert read.metadata["graph"] == written.metadata["graph"]
        assert LaggedGraph.from_dict(read.metadata["graph"]) == LaggedGraph.from_dict(
            written.metadata["graph"]
        )


def test_constructor_normalizes_and_validates():
    graph = LaggedGraph(
        n=np.int64(2),
        columns=["a", "b"],
        edges=[[np.int64(0), np.int64(1), np.int64(1)], (0, 1, 1)],
        k_sampled=np.int32(1),
        k_eff=1,
        hidden=[np.int64(1)],
    )
    assert graph.edges == ((0, 1, 1),)
    assert type(graph.edges[0][0]) is int
    as_dict = graph.to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert LaggedGraph.from_dict({**as_dict, "path": []}) == graph
    base = {"n": 2, "columns": ("a", "b"), "edges": ((0, 1, 1),), "k_sampled": 1, "k_eff": 1}
    bad = [
        ({"n": 3}, ValueError, "column names"),
        ({"columns": ("a", "a")}, ValueError, "repeat"),
        ({"edges": ((0, 2, 1),)}, ValueError, "outside"),
        ({"edges": ((0, 1, -1),)}, ValueError, "negative"),
        ({"edges": ((0, 1),)}, ValueError, "triple"),
        ({"edges": ((0, 1, 1.5),)}, TypeError, "integer"),
        ({"k_eff": 2}, ValueError, "k_eff"),
        ({"k_sampled": 0}, ValueError, "k_sampled"),
        ({"hidden": (2,)}, ValueError, "hidden"),
        ({"reads_parents": "false"}, TypeError, "reads_parents"),
        ({"time": "hourly"}, ValueError, "time"),
        ({"columns": ("a", 1)}, TypeError, "non-string"),
    ]
    for change, error, match in bad:
        with pytest.raises(error, match=match):
            LaggedGraph(**{**base, **change})
    with pytest.raises(KeyError, match="edges"):
        LaggedGraph.from_dict({"n": 1, "columns": ["a"], "k_sampled": 0, "k_eff": 0})


@pytest.mark.parametrize("compress", [False, True])
def test_load_graph_sidecar(tmp_path, compress):
    graph = LaggedGraph.from_structure("mediator")
    records = [{"idx": i, "graph": graph.to_dict(), "min_lag": 1} for i in (3, 1)]
    text = "".join(json.dumps(r) + "\n" for r in records)
    path = tmp_path / "sidecar.jsonl"
    if compress:
        # A gzip stream under a plain name: the loader reads the magic bytes.
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text + "\n")
    loaded = load_graph_sidecar(path)
    assert sorted(loaded) == [1, 3]
    assert loaded[3]["graph"] == graph
    assert loaded[1]["min_lag"] == 1
    path.write_text(text + text)
    with pytest.raises(ValueError, match="twice"):
        load_graph_sidecar(path)


@pytest.mark.parametrize("suite", list(_SIDECARS))
def test_published_sidecar_matches_a_fresh_regeneration(suite):
    """The first record of each SCM family must equal a regenerated one.

    Ties the published sidecars to the code: a change to the edge convention
    or to the path summary shows up here, not in a reviewer's analysis.
    """
    yaml = pytest.importorskip("yaml")
    path = _REPO / "results" / "reference" / _SIDECARS[suite]
    if not path.exists():
        pytest.skip(f"{path.name} is not in this checkout")
    config = yaml.safe_load((_REPO / "scripts" / "release_config.yaml").read_text())
    position = list(config["suites"]).index(suite)
    seed = int(config["seed"]) + 1000 * (position + 1)  # build_release.py's suite seed
    specs = episode_specs(config["suites"][suite], seed, 1.0)
    first: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            first.setdefault(record["family"], record)
            if len(first) == (1 if suite == "dot-RegimeSwitch-v1" else 3):
                break
    for record in first.values():
        assert record["verified"]
        ep = make_episode({**specs[record["idx"]], "record_graph": True})
        stored = dict(ep.metadata["graph"])
        (entry,) = stored.pop("path")
        assert stored == record["graph"]
        assert entry == {
            "target": record["query"],
            **{k: record[k] for k in ("reachable", "min_lag", "min_hops", "direct_lags")},
        }
        assert record["treatment"] == ep.intervention.targets
        assert (record["onset"], record["window_end"]) == (
            min(ep.intervention.times),
            max(ep.intervention.times),
        )


@pytest.mark.parametrize(
    "label",
    [
        "bi_variate+seasonal_observed",
        "bi_variate+trend_hidden",
        "back_door+seasonal_observed",
        "back_door+trend_hidden",
    ],
)
def test_from_structure_handles_driven_labels(label):
    """A driven label adds the root ``D`` at column ``N - 2`` and matches the built episode."""
    base = LaggedGraph.from_structure(label.split("+")[0])
    graph = LaggedGraph.from_structure(label)
    assert graph.n == base.n + 1
    d = graph.columns.index("D")
    assert d == graph.n - 2
    assert graph.columns[0] == "A"
    assert graph.columns[-1] == "Y"
    assert (d, 0, 0) in graph.edges
    assert (d, graph.n - 1, 0) in graph.edges
    assert all(lag == 0 for src, dst, lag in graph.edges if d in (src, dst))
    assert (d in graph.hidden) == label.endswith("_hidden")
    # The base edges survive with their columns shifted around the inserted D.
    assert len(graph.edges) == len(base.edges) + 2
    spec = {
        "generator": "identifiability",
        "kind": "identifiability",
        "structure": label,
        "T": 60,
        "seed": 3,
        "idx": 0,
        "tier": 2,
        "pair_mode": "counterfactual",
        "record_graph": True,
    }
    ep = make_episode(spec)
    recorded = LaggedGraph.from_dict(ep.metadata["graph"])
    assert recorded.columns == graph.columns
    assert recorded.hidden == graph.hidden
    assert ep.metadata["driver"]["column"] == d
    assert ep.x_obs.shape[1] == graph.n
