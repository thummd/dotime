"""Command-line interface for DoTime.

Two console scripts are installed with the package (see ``[project.scripts]`` in
``pyproject.toml``):

    dotime-generate    sample paired trajectories from the prior and write to disk
    dotime-benchmark   evaluate a baseline against a frozen benchmark suite

Run ``dotime-generate --help`` or ``dotime-benchmark --help`` for usage.

The functions :func:`generate_main` and :func:`benchmark_main` are the entry
points; both accept an optional ``argv`` list (handy for tests) and return a
process exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotime import __version__

_AVAILABLE_SUITES = (
    "CTP-Identifiability-v1",
    "CTP-RegimeSwitch-v1",
    "CTP-Continuous-v1",
    "CTP-Generic-100k",
)

_INTERVENTION_SOURCES = (
    "prior",
    "observed_discrete",
    "observed_normal",
    "observed_uniform",
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device: cpu, cuda, cuda:0, ... (default: cpu).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stderr.")
    parser.add_argument("--version", action="version", version=f"dotime {__version__}")


# --------------------------------------------------------------------------- #
# dotime-generate
# --------------------------------------------------------------------------- #


def _build_generate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dotime-generate",
        description="Sample paired observational/interventional trajectories from the prior.",
    )
    p.add_argument(
        "-n", "--n-scms", type=int, default=1000, help="Number of SCMs to sample (default: 1000)."
    )
    p.add_argument(
        "-T", "--length", type=int, default=200, help="Trajectory length T (default: 200)."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file. Extension selects the format: .pt (torch) or .parquet.",
    )
    p.add_argument(
        "--intervention-source",
        choices=_INTERVENTION_SOURCES,
        default="prior",
        help="Counterfactual sampling mode for intervention values (default: prior).",
    )
    _add_common(p)
    return p


def generate_main(argv: list[str] | None = None) -> int:
    """Entry point for ``dotime-generate``."""
    args = _build_generate_parser().parse_args(argv)

    from dotime import DoTime

    if args.verbose:
        print(
            f"[dotime-generate] sampling {args.n_scms} SCMs at T={args.length} "
            f"(seed={args.seed}, source={args.intervention_source})",
            file=sys.stderr,
        )

    prior = DoTime(seed=args.seed)
    dataset = prior.generate_dataset(n_scms=args.n_scms, T=args.length)

    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".pt":
        import torch

        torch.save(dataset, out)
    elif out.suffix == ".parquet":
        _write_parquet(dataset, out, seed=args.seed)
    else:
        raise SystemExit(
            f"error: unsupported output extension {out.suffix!r} (use .pt or .parquet)"
        )

    print(f"[dotime-generate] wrote {len(dataset)} trajectories to {out}")
    return 0


def _write_parquet(dataset: list, out: Path, *, seed: int) -> None:
    """Serialize a generated dataset to the frozen-suite parquet schema.

    Each ``(X_obs, X_int, intervention)`` tuple becomes one Episode (with a query
    derived from the most intervention-affected variable), written via the
    canonical :mod:`dotime._release_io` schema. ``out``'s ``.parquet``
    suffix is treated as the suite-directory stem.
    """
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "parquet output requires the 'evaluation' extra:\n"
            "    pip install 'dotime[evaluation]'"
        ) from exc

    from dotime import __version__, _release_io
    from dotime.benchmarks import SuiteMetadata, episode_from_pair

    episodes = [
        episode_from_pair(x_obs, x_int, intervention, scm_id=i)
        for i, (x_obs, x_int, intervention) in enumerate(dataset)
    ]
    suite_dir = out.with_suffix("") if out.suffix == ".parquet" else out
    meta = SuiteMetadata(
        name=suite_dir.name,
        version="0.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="Locally generated via dotime-generate.",
        n_episodes=len(episodes),
    )
    _release_io.write_suite(meta, episodes, suite_dir, package_version=__version__, seed=seed)


# --------------------------------------------------------------------------- #
# dotime-benchmark
# --------------------------------------------------------------------------- #


def _build_benchmark_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dotime-benchmark",
        description="Evaluate a baseline against a frozen DoTime benchmark suite.",
    )
    p.add_argument("--list", action="store_true", help="List available benchmark suites and exit.")
    p.add_argument(
        "--list-baselines", action="store_true", help="List available baselines and exit."
    )
    p.add_argument(
        "--suite",
        default=None,
        help="Suite name to evaluate against, e.g. CTP-Identifiability-v1.",
    )
    p.add_argument(
        "--baseline",
        default="VAR-OLS",
        help="Baseline name (see --list-baselines). Default: VAR-OLS.",
    )
    p.add_argument(
        "--json-out", type=Path, default=None, help="Write the full results dict to this JSON path."
    )
    _add_common(p)
    return p


def benchmark_main(argv: list[str] | None = None) -> int:
    """Entry point for ``dotime-benchmark``."""
    args = _build_benchmark_parser().parse_args(argv)

    if args.list:
        print("Available benchmark suites:")
        for suite_name in _AVAILABLE_SUITES:
            print(f"  {suite_name}")
        return 0

    # Imported lazily so `dotime-benchmark --list` works without the baselines extra.
    from dotime import baselines as _baselines

    if args.list_baselines:
        print("Available baselines:")
        # TODO(api): expose `baselines.available()` returning the registry keys.
        for name in getattr(_baselines, "available", lambda: [])():
            print(f"  {name}")
        return 0

    if args.suite is None:
        raise SystemExit("error: --suite is required (or pass --list). See --help.")
    if args.suite not in _AVAILABLE_SUITES:
        raise SystemExit(
            f"error: unknown suite {args.suite!r}. Run `dotime-benchmark --list` to see options."
        )

    from dotime.benchmarks import load_benchmark
    from dotime.evaluation import evaluate

    if args.verbose:
        print(f"[dotime-benchmark] loading suite {args.suite}", file=sys.stderr)

    suite = load_benchmark(args.suite)

    # TODO(api): expose `baselines.get(name)` returning an instantiated baseline.
    model = _baselines.get(args.baseline)
    results = evaluate(model, suite)

    # Results objects are expected to provide `.summary()` (human-readable) and
    # `.to_dict()` (serializable); fall back to repr if not yet implemented.
    print(results.summary() if hasattr(results, "summary") else results)

    if args.json_out is not None:
        import json

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = results.to_dict() if hasattr(results, "to_dict") else results
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"[dotime-benchmark] wrote results to {args.json_out}")

    return 0


# --------------------------------------------------------------------------- #
# Allow `python -m dotime.cli generate ...` style invocation too.
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``python -m dotime.cli {generate,benchmark} ...``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m dotime.cli {generate,benchmark} [options]")
        return 0 if argv else 1
    sub, rest = argv[0], argv[1:]
    if sub == "generate":
        return generate_main(rest)
    if sub == "benchmark":
        return benchmark_main(rest)
    raise SystemExit(f"error: unknown subcommand {sub!r} (expected 'generate' or 'benchmark')")


if __name__ == "__main__":
    raise SystemExit(main())
