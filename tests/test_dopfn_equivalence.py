"""Equivalence test: reimplemented Do-PFN surface vs. the original ``dopfnprior``.

Phase 2a of the consolidation reimplemented the small ``Do-PFN-prior`` surface
(samplers, the random-DAG builder, the linear mechanism, and the non-standard
activations) as first-class, attributed modules so the package carries no git
submodule. This test pins seeds and asserts the reimplementation reproduces the
original bit-for-bit on the RNG-determining paths — without which "reimplemented
with credit" would silently break reproducibility of the published TSALM / FMSD
numbers.

The original ``dopfnprior`` is an optional, test-only dependency: the test is
skipped (not failed) when it is not importable, so it never blocks a clean
install. Run it in an environment where ``dopfnprior`` is present (e.g. the
development env that produced the paper numbers).
"""

from __future__ import annotations

import math

import pytest
import torch

dopfnprior = pytest.importorskip(
    "dopfnprior", reason="original Do-PFN-prior not installed; equivalence test skipped"
)

from dotime import _activations, _graph_builder, _mechanism, _sampling


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #


def test_torch_distribution_sampler_matches():
    import torch.distributions as dist
    from dopfnprior.utils.sampling import TorchDistributionSampler as Orig

    ours = _sampling.TorchDistributionSampler(dist.Normal(0.0, 1.0))
    theirs = Orig(dist.Normal(0.0, 1.0))
    a = ours.sample_n(1000, _gen(123))
    b = theirs.sample_n(1000, _gen(123))
    assert torch.equal(a, b)


def test_shifted_exponential_matches():
    from dopfnprior.utils.sampling import ShiftedExponentialSampler as Orig

    ours = _sampling.ShiftedExponentialSampler(rate=2.0, shift=0.5)
    theirs = Orig(rate=2.0, shift=0.5)
    a = ours.sample_n(500, _gen(7))
    b = theirs.sample_n(500, _gen(7))
    assert torch.equal(a, b)


def test_discrete_uniform_matches():
    from dopfnprior.utils.sampling import DiscreteUniformSampler as Orig

    ours = _sampling.DiscreteUniformSampler(3, 9)
    theirs = Orig(3, 9)
    a = ours.sample_n(500, _gen(11))
    b = theirs.sample_n(500, _gen(11))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #


def test_activations_match():
    from dopfnprior.configs.default_config import Tanh, TanhReLU, TanhX2

    x = torch.linspace(-3, 3, 101)
    assert torch.allclose(_activations.Tanh()(x), Tanh()(x))
    assert torch.allclose(_activations.TanhX2()(x), TanhX2()(x))
    assert torch.allclose(_activations.TanhReLU()(x), TanhReLU()(x))


# --------------------------------------------------------------------------- #
# Mechanism
# --------------------------------------------------------------------------- #


def test_simple_mechanism_init_and_forward_match():
    from dopfnprior.scm.simple_mechanism import SimpleMechanism as Orig

    nodes = ["x0", "x1", "x2"]
    device = torch.device("cpu")

    ours = _mechanism.SimpleMechanism(nodes, _activations.Tanh(), device, _gen(42))
    theirs = Orig(nodes, _activations.Tanh(), device, _gen(42))

    # Same weight/bias initialisation under the same generator.
    for v in nodes:
        assert torch.equal(ours.weights[v].detach(), theirs.weights[v].detach())
    assert torch.equal(ours.bias.detach(), theirs.bias.detach())

    parents = {v: torch.randn(5, generator=_gen(i)) for i, v in enumerate(nodes)}
    eps = torch.randn(5, generator=_gen(99))
    assert torch.allclose(ours(parents, eps), theirs(parents, eps))


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
def test_graph_builder_matches(seed):
    from dopfnprior.causal_graph.graph_builder import GraphBuilder as Orig

    ours = _graph_builder.GraphBuilder(num_nodes=8, edge_prob=0.3, dropout_prob=0.1)
    theirs = Orig(num_nodes=8, edge_prob=0.3, dropout_prob=0.1)

    g_ours = ours.sample(_gen(seed))
    g_theirs = theirs.sample(_gen(seed))

    assert set(g_ours.nodes) == set(g_theirs.nodes)
    assert set(g_ours.edges) == set(g_theirs.edges)


def test_edge_prob_floor_matches():
    # The min-edge-probability floor (2 / n**1.2) must match exactly.
    for n in (2, 3, 5, 10, 20):
        ours = _graph_builder.GraphBuilder(num_nodes=n, edge_prob=0.0, dropout_prob=0.0)
        assert math.isclose(ours.edge_prob, 2 / (n**1.2))
