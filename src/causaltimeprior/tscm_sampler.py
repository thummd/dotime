"""Targeted Temporal SCM Sampler for identifiability case studies.

Generates specific causal structures (confounder, mediator, etc.)
with random mechanisms and noise, enabling systematic identifiability
analysis of the trained model.

Variable naming convention (consistent across all structures):
  A  - treatment (always the intervention target)
  Y  - outcome   (always the query target)
  M  - mediator
  X  - observed confounder or instrument
  U  - unobserved (hidden) confounder
"""

from enum import Enum

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from causaltimeprior._activations import Tanh, TanhReLU
from causaltimeprior._sampling import TorchDistributionSampler
from causaltimeprior.temporal_graph import TemporalDAG
from causaltimeprior.temporal_mechanism import TemporalMechanism
from causaltimeprior.temporal_scm import TemporalSCM


class TSCMStructure(Enum):
    """Named temporal causal structures for identifiability studies.

    Identification strategies (cf. identifiability.tex):
    - Backdoor: OBSERVED_CONFOUNDER, BACK_DOOR, CONFOUNDER_MEDIATOR
      Adjust via observed covariates that block all back-door paths.
    - Frontdoor: FRONT_DOOR, MEDIATOR
      Adjust via mediator when confounder is hidden.
    - Trivially identified: RCT_NO_CONFOUNDING
      No confounding => p(Y|do(A)) = p(Y|A).
    - IV: INSTRUMENTAL_VARIABLE
      X -> A -> Y with hidden confounding U -> A, U -> Y.
    - Non-identifiable: UNOBSERVED_CONFOUNDER
      Hidden confounder, no mediator or instrument. Tests model robustness.
    """

    OBSERVED_CONFOUNDER = "observed_confounder"  # X -> A, X -> Y (backdoor)
    MEDIATOR = "mediator"  # A -> M -> Y (frontdoor, trivial)
    CONFOUNDER_MEDIATOR = "confounder_mediator"  # X -> A -> M -> Y, X -> Y (backdoor)
    UNOBSERVED_CONFOUNDER = "unobserved_confounder"  # U -> A, U -> Y (non-identifiable)
    BACK_DOOR = "back_door"  # X -> A, X -> Y, A -> Y (backdoor)
    FRONT_DOOR = "front_door"  # A -> M -> Y, U -> A, U -> Y (frontdoor)
    INSTRUMENTAL_VARIABLE = "instrumental_variable"  # X -> A -> Y, U -> A, U -> Y (IV)
    RCT_NO_CONFOUNDING = "rct_no_confounding"  # A -> Y (trivially identified)


# Default activations for random mechanism sampling
DEFAULT_ACTIVATIONS = [
    nn.Identity(),
    Tanh(),
    TanhReLU(),
    nn.ReLU(),
]


class TSCMSampler:
    """Samples temporal SCMs with specific causal structures.

    Unlike CausalTimePrior which randomly samples graph structure,
    this builds a fixed named structure with random mechanisms/noise.

    All structures use the convention:
      A = treatment (intervention target)
      Y = outcome   (query target)
      M = mediator
      X = observed confounder / instrument
      U = unobserved confounder (hidden, index 0 when present)
    """

    def __init__(
        self,
        structure: TSCMStructure,
        max_lag: int = 1,
        activations: list[nn.Module] | None = None,
        sigma_w: float = 1.0,
        sigma_b: float = 0.5,
        use_lagged_edges: bool = True,
        noise_std: float = 0.3,
        device: str = "cpu",
    ):
        self.structure = structure
        self.max_lag = max_lag
        self.activations = activations or DEFAULT_ACTIVATIONS
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.noise_std = noise_std
        self.device = torch.device(device)
        self.use_lagged_edges = use_lagged_edges

    def sample(self, generator: torch.Generator | None = None) -> TemporalSCM:
        """Sample a temporal SCM with the specified structure.

        Returns
        -------
        TemporalSCM with the fixed structure but random mechanisms/noise.
        """
        dag = self._build_dag()
        node_names = dag.topo_order
        mechanisms = self._sample_mechanisms(node_names, dag, generator)
        noise = self._sample_noise(node_names)
        return TemporalSCM(dag, mechanisms, noise, device=self.device)

    def get_hidden_vars(self) -> list[int]:
        """Return indices of hidden (unobserved) variables, if any.

        U is always placed first (index 0) when present.
        """
        if self.structure in (
            TSCMStructure.UNOBSERVED_CONFOUNDER,
            TSCMStructure.FRONT_DOOR,
            TSCMStructure.INSTRUMENTAL_VARIABLE,
        ):
            return [0]  # U is always index 0
        return []

    def get_outcome_var(self) -> int:
        """Return the topological-order index of the outcome variable Y."""
        dag = self._build_dag()
        return dag.topo_order.index("Y")

    def get_intervention_target(self) -> int:
        """Return the topological-order index of the treatment variable A.

        Use this instead of valid_targets[0] in evaluation code to ensure
        the correct variable is intervened upon for each structure.
        """
        dag = self._build_dag()
        return dag.topo_order.index("A")

    def _build_dag(self) -> TemporalDAG:
        """Construct a TemporalDAG for the specified structure."""
        builder = getattr(self, f"_build_{self.structure.value}")
        return builder()

    # --- Structure builders ---

    def _build_observed_confounder(self) -> TemporalDAG:
        """X -> A (inst), X(t-1) -> Y(t) (lagged). Backdoor via observed X.

        Topo order: ['X', 'Y', 'A']  (Y has no G_0 parents, A has parent X).
        G_lags indices follow topo order: X=0, Y=1, A=2.
        """
        G_0 = nx.DiGraph()
        nodes = ["X", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("X", "A")  # instantaneous: confounder -> treatment
        # X -> Y via lag only (no instantaneous path from A to Y)

        # Build the lag matrix using topo-order indices, not insertion indices.
        # nx.topological_sort gives ['X', 'Y', 'A'] for this graph, so:
        #   topo index 0 = X,  1 = Y,  2 = A
        topo = list(nx.topological_sort(G_0))
        x_i = topo.index("X")
        y_i = topo.index("Y")

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:  # lag-1
                G_k[x_i, y_i] = 1.0  # X(t-1) -> Y(t)
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_mediator(self) -> TemporalDAG:
        """A(t-1) -> M(t) (lagged), M -> Y (inst). Front-door via mediator M."""
        G_0 = nx.DiGraph()
        nodes = ["A", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("M", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                G_k[0, 1] = 1.0  # A(t-1) -> M(t)
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_confounder_mediator(self) -> TemporalDAG:
        """X -> A -> M -> Y (inst), X(t-1) -> Y(t) (lagged). Backdoor + mediator."""
        G_0 = nx.DiGraph()
        nodes = ["X", "A", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("X", "A")  # instantaneous
        G_0.add_edge("A", "M")  # instantaneous
        G_0.add_edge("M", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                G_k[0, 3] = 1.0  # X(t-1) -> Y(t)
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_unobserved_confounder(self) -> TemporalDAG:
        """U -> A, U -> Y (inst). U hidden (index 0). Non-identifiable."""
        G_0 = nx.DiGraph()
        nodes = ["U", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "A")  # instantaneous
        G_0.add_edge("U", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_back_door(self) -> TemporalDAG:
        """X -> A, A -> Y (inst), X -> Y (inst), X(t-1) -> Y(t) (lagged if enabled).

        X = confounder, A = treatment, Y = outcome.
        X satisfies the back-door criterion. When use_lagged_edges=True,
        X(t-1)->Y(t) is added as a lagged edge making confounding observable.
        """
        G_0 = nx.DiGraph()
        nodes = ["X", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("X", "A")  # confounder -> treatment
        G_0.add_edge("X", "Y")  # confounder -> outcome (instantaneous)
        G_0.add_edge("A", "Y")  # treatment -> outcome

        topo = list(nx.topological_sort(G_0))
        x_i = topo.index("X")
        y_i = topo.index("Y")

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                if self.use_lagged_edges:
                    G_k[x_i, y_i] = 1.0  # X(t-1) -> Y(t): lagged confounding
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_front_door(self) -> TemporalDAG:
        """A -> M (inst), M -> Y (inst), U -> A, U -> Y (inst). M(t-1)->Y(t) lagged if enabled.

        U = hidden confounder, A = treatment, M = mediator, Y = outcome.
        M satisfies the front-door criterion. When use_lagged_edges=True,
        M(t-1)->Y(t) is added making mediation observable from history.
        """
        G_0 = nx.DiGraph()
        nodes = ["U", "A", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "A")  # hidden confounder -> treatment
        G_0.add_edge("U", "Y")  # hidden confounder -> outcome
        G_0.add_edge("A", "M")  # treatment -> mediator
        G_0.add_edge("M", "Y")  # mediator -> outcome

        topo = list(nx.topological_sort(G_0))
        m_i = topo.index("M")
        y_i = topo.index("Y")

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                if self.use_lagged_edges:
                    G_k[m_i, y_i] = 1.0  # M(t-1) -> Y(t): lagged mediation
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_instrumental_variable(self) -> TemporalDAG:
        """X -> A -> Y (inst), U -> A, U -> Y (inst). U hidden. X is instrument."""
        G_0 = nx.DiGraph()
        nodes = ["U", "X", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "A")  # hidden confounder -> treatment
        G_0.add_edge("U", "Y")  # hidden confounder -> outcome
        G_0.add_edge("X", "A")  # instrument -> treatment
        G_0.add_edge("A", "Y")  # treatment -> outcome

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_rct_no_confounding(self) -> TemporalDAG:
        """A -> Y (inst). No confounding. Trivially identified: p(Y|do(A)) = p(Y|A)."""
        G_0 = nx.DiGraph()
        nodes = ["A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("A", "Y")  # direct causal effect

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    # --- Mechanism/noise sampling ---

    def _sample_mechanisms(
        self,
        node_names: list[str],
        dag: TemporalDAG,
        generator: torch.Generator | None,
    ) -> dict:
        """Sample random mechanisms for each node."""
        mechanisms = {}
        for v in node_names:
            # Pick random activation
            act_idx = int(torch.randint(0, len(self.activations), (1,), generator=generator).item())
            activation = self.activations[act_idx]

            mechanisms[v] = TemporalMechanism(
                node_names=node_names,
                activation=activation,
                num_lags=dag.K,
                device=self.device,
                generator=generator,
                sigma_w=self.sigma_w,
                sigma_b=self.sigma_b,
            )
        return mechanisms

    def _sample_noise(self, node_names: list[str]) -> dict:
        """Create noise distributions for each node."""
        noise = {}
        for v in node_names:
            noise[v] = TorchDistributionSampler(torch.distributions.Normal(0.0, self.noise_std))
        return noise
