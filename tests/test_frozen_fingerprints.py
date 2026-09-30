"""Released suites must keep regenerating: pinned rows of every frozen suite version.

``tests/data/frozen_fingerprints.json`` holds about 25 stratified rows of each
released suite version (Identifiability 1.0.0 and 1.1.0, RegimeSwitch,
Continuous and Generic 1.0.0), recorded by
``scripts/fingerprint_frozen_suites.py`` from the md5-verified release files.
Each row is regenerated here with the current package. Its portable summary
(shapes, intervention, query, RNG-only metadata) must match on every platform.
Its exact column hashes must match only where they were recorded: float
results differ in the last bit between CPUs and library builds, and CI runs
macOS arm64 as well as x86 runners. ``DOTIME_FINGERPRINT_EXACT=1`` or ``=0``
forces the exact check on or off.

A failure here means a default code path no longer regenerates a released
suite. If the change is intended, rerun the script and review the diff.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from dotime import _release_io
from dotime._build import episode_specs, make_episode
from dotime._fingerprint import portable_summary, reference_env, row_hashes, summary_mismatches

_ROOT = Path(__file__).resolve().parents[1]
_DATA = json.loads((_ROOT / "tests" / "data" / "frozen_fingerprints.json").read_text())
_REGENERATE = "python scripts/fingerprint_frozen_suites.py"

# The v1 row schema, spelled out so that a change to _release_io._COLUMNS
# fails here instead of silently re-pinning.
V1_COLUMNS = (
    "scm_id",
    "structure",
    "tier",
    "n_vars",
    "length",
    "x_obs",
    "x_int",
    "intervention_json",
    "query_target",
    "query_time",
    "y_true",
    "metadata_json",
)

# Columns that may differ from the release, and the versions where they may.
_DOCUMENTED = {
    "metadata_json": None,
    "x_obs": {"dot-Identifiability-v1-1.0.0"},
    "structure": {"dot-Identifiability-v1-1.0.0"},
}


def _exact() -> bool:
    """Whether to compare exact row hashes in this environment.

    Returns:
        The ``DOTIME_FINGERPRINT_EXACT`` override when set, else whether this
        environment equals the recorded ``reference_env``.
    """
    flag = os.environ.get("DOTIME_FINGERPRINT_EXACT")
    if flag is None:
        return reference_env() == _DATA["reference_env"]
    if flag not in ("0", "1"):
        pytest.fail(f"DOTIME_FINGERPRINT_EXACT must be 0 or 1, got {flag!r}")
    return flag == "1"


def test_fingerprints_cover_every_released_version() -> None:
    """Every frozen version is pinned, with the v1 columns and only documented differences."""
    assert sorted(_DATA["versions"]) == [
        "dot-Continuous-v1-1.0.0",
        "dot-Generic-100k-1.0.0",
        "dot-Identifiability-v1-1.0.0",
        "dot-Identifiability-v1-1.1.0",
        "dot-RegimeSwitch-v1-1.0.0",
    ]
    assert tuple(_DATA["columns"]) == V1_COLUMNS
    for key, version in _DATA["versions"].items():
        assert len(version["rows"]) >= 20, key
        for row in version["rows"]:
            assert tuple(row["hashes"]) == V1_COLUMNS
            for col, ok in row["release_match"].items():
                versions = _DOCUMENTED.get(col, set())
                allowed = versions is None or key in versions
                assert ok or allowed, f"{key} row {row['idx']}: undocumented {col} difference"


def test_specs_rebuilt_from_the_release_configs_equal_the_pinned_ones() -> None:
    """The committed YAMLs and build_release's seed rule still describe the releases.

    The pinned seeds were read from the release manifests, so this also checks
    the derivation base seed + 1000 * (position + 1).
    """
    for key, version in _DATA["versions"].items():
        cfg_all = yaml.safe_load((_ROOT / "scripts" / version["config"]).read_text())
        names = list(cfg_all["suites"])
        seed = int(cfg_all["seed"]) + 1000 * (names.index(version["suite"]) + 1)
        assert seed == version["suite_seed"], key
        cfg = cfg_all["suites"][version["suite"]]
        assert cfg["version"] == version["version"], key
        specs = episode_specs(cfg, seed, 1.0)
        assert len(specs) == version["n_episodes"], key
        for row in version["rows"]:
            assert json.loads(json.dumps(specs[row["idx"]])) == row["spec"], (key, row["idx"])


@pytest.mark.parametrize("key", sorted(_DATA["versions"]))
def test_pinned_rows_regenerate(key: str) -> None:
    """Regenerate every pinned row: portable summaries always, exact hashes where recorded.

    Args:
        key: Suite version, e.g. ``"dot-Continuous-v1-1.0.0"``.
    """
    exact = _exact()
    summary_problems: list[str] = []
    hash_problems: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # handled SCM divergence
        for row in _DATA["versions"][key]["rows"]:
            ep = make_episode(row["spec"])
            cols = _release_io._episode_to_row(ep)
            assert tuple(cols) == V1_COLUMNS, f"{key} row {row['idx']}: columns {tuple(cols)}"
            summary_problems += summary_mismatches(
                row["portable"], portable_summary(ep), path=f"row {row['idx']}"
            )
            if exact:
                hashes = row_hashes(cols)
                hash_problems += [
                    f"row {row['idx']} {col}"
                    for col in V1_COLUMNS
                    if hashes[col] != row["hashes"][col]
                ]
    assert not summary_problems, (
        f"{key}: regenerated rows no longer match their portable summaries "
        f"(rerun `{_REGENERATE}` only if intended):\n" + "\n".join(summary_problems[:20])
    )
    assert not hash_problems, (
        f"{key}: regenerated columns differ from the pinned hashes in the reference "
        f"environment (rerun `{_REGENERATE}` only if intended): {hash_problems[:20]}"
    )


def test_row_hashes_are_type_driven_and_match_a_parquet_round_trip(tmp_path: Path) -> None:
    """A row written to parquet and read back hashes exactly like the row in memory."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row = _release_io._episode_to_row(
            make_episode({"kind": "generic", "idx": 0, "seed": 7, "T": 40})
        )
    pq.write_table(pa.table({c: [row[c]] for c in row}), tmp_path / "row.parquet")
    table = pq.read_table(tmp_path / "row.parquet")
    back = {c: table.column(c)[0].as_py() for c in table.column_names}
    assert row_hashes(back) == row_hashes(row)
    # Integer lists and float lists with equal values must not collide.
    assert row_hashes({"a": [1, 2]})["a"] != row_hashes({"a": [1.0, 2.0]})["a"]
    with pytest.raises(TypeError, match="flag"):
        row_hashes({"flag": True})


def test_summary_mismatches_tolerates_rounding_only() -> None:
    """Floats may differ within the tolerance; integers, keys and strings may not."""
    stored = {"a": [1.234568, 2], "b": "x", "c": {"d": None}}
    assert summary_mismatches(stored, {"a": [1.2345678, 2], "b": "x", "c": {"d": None}}) == []
    assert summary_mismatches(stored, {"a": [1.3, 2], "b": "x", "c": {"d": None}})
    assert summary_mismatches(stored, {"a": [1.234568, 3], "b": "x", "c": {"d": None}})
    assert summary_mismatches(stored, {"a": [1.234568, 2], "b": "y", "c": {"d": None}})
    assert summary_mismatches(stored, {"a": [1.234568, 2], "b": "x", "c": {}})
    assert summary_mismatches({"n": 1}, {"n": True}) == ["summary.n: 1 != True"]
