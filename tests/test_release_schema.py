"""Suite schema 2: optional observation times and masks beside the v1 columns.

The frozen suites are schema 1, and readers match schema versions exactly, so
schema 2 has to leave every schema-1 file byte-identical and stay readable next
to it. The reference writer below is a frozen copy of the schema-1 writer as it
was released, so the byte test does not reuse the code under test.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyarrow", reason="frozen-suite IO needs the evaluation extra")
import pyarrow as pa
import pyarrow.parquet as pq

from dotime import DoTime, _release_io
from dotime.benchmarks import SuiteMetadata, episode_from_pair, episode_from_sample
from dotime.continuous import ContinuousExtendedPrior
from dotime.evaluation import realign_episode

_V1_COLUMNS = (
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


def _meta(name: str = "S") -> SuiteMetadata:
    return SuiteMetadata(
        name=name, version="1.0.0", zenodo_record_id="LOCAL", doi="", description="", n_episodes=0
    )


def _episodes(n: int = 4, t_len: int = 40) -> list:
    """Finite, fully observed episodes of both constructor kinds."""
    torch.manual_seed(5)
    prior = DoTime(seed=5)
    eps = []
    for i in range(n):
        x_obs, x_int, iv, _ = prior.generate_pair(T=t_len)
        eps.append(episode_from_pair(x_obs, x_int, iv, structure="back_door", scm_id=i))
    cont = ContinuousExtendedPrior(tscm_structure="front_door", seed=9)
    for i in range(2):
        s = cont.generate_sample(T=t_len)
        eps.append(
            episode_from_sample(s, structure="front_door", scm_id=n + i, metadata={"tier": 2})
        )
    return eps


def _v1_row(ep) -> dict:
    """A schema-1 row exactly as the released writer built it."""

    def jsonable(v):
        return v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v

    meta = {k: jsonable(v) for k, v in ep.metadata.items() if k != "y_oracle"}
    return {
        "scm_id": int(ep.scm_id if ep.scm_id is not None else -1),
        "structure": ep.structure or "",
        "tier": int(ep.metadata.get("tier", 0)),
        "n_vars": int(ep.x_obs.shape[1]),
        "length": int(ep.x_obs.shape[0]),
        "x_obs": ep.x_obs.detach().cpu().reshape(-1).tolist(),
        "x_int": ep.x_int.detach().cpu().reshape(-1).tolist(),
        "intervention_json": json.dumps(ep.intervention.to_dict()),
        "query_target": ep.query_target.detach().cpu().reshape(-1).tolist(),
        "query_time": ep.query_time.detach().cpu().reshape(-1).tolist(),
        "y_true": ep.y_true.detach().cpu().reshape(-1).tolist(),
        "metadata_json": json.dumps(meta),
    }


def _write_v1(meta, episodes, dest: Path, *, seed: int, shard_size: int, extra: dict) -> Path:
    """The released schema-1 writer: inferred column types, manifest schema "1"."""
    dest.mkdir(parents=True)
    shards = []
    for shard_idx, start in enumerate(range(0, len(episodes), shard_size)):
        chunk = episodes[start : start + shard_size]
        rows = [_v1_row(ep) for ep in chunk]
        table = pa.table({col: [r[col] for r in rows] for col in _V1_COLUMNS})
        fname = f"shard-{shard_idx:04d}.parquet"
        pq.write_table(table, dest / fname)
        shards.append(
            {"file": fname, "n_episodes": len(chunk), "md5": _release_io._md5(dest / fname)}
        )
    manifest = {
        "name": meta.name,
        "version": meta.version,
        "schema_version": "1",
        "package_version": "0.1.0",
        "seed": seed,
        "n_episodes": len(episodes),
        "structures": list(meta.structures),
        "license": meta.license,
        "shards": shards,
        **extra,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def test_schema_1_suites_stay_byte_identical(tmp_path):
    eps, meta, extra = _episodes(), _meta(), {"generator": "mixed", "scheme": "perepisode"}
    new = _release_io.write_suite(
        meta,
        eps,
        tmp_path / "new",
        package_version="0.1.0",
        seed=3,
        shard_size=4,
        extra_manifest=extra,
    )
    old = _write_v1(meta, eps, tmp_path / "old", seed=3, shard_size=4, extra=extra)
    names = sorted(p.name for p in old.iterdir())
    assert names == ["manifest.json", "shard-0000.parquet", "shard-0001.parquet"]
    assert sorted(p.name for p in new.iterdir()) == names
    for name in names:
        assert _release_io._md5(new / name) == _release_io._md5(old / name), name
    # The row dict itself is unchanged too (key order included), since
    # fingerprints of the release hash it.
    for ep in eps:
        row = _release_io._episode_to_row(ep)
        assert list(row) == list(_V1_COLUMNS)
        assert row == _v1_row(ep)


def _schema_2_episodes() -> list:
    base = _episodes(n=5)
    nan_obs = base[0].x_obs.clone()
    nan_obs[3, 1] = float("nan")
    nan_obs[7:9, 0] = float("nan")
    explicit = torch.ones_like(base[1].x_obs, dtype=torch.bool)
    explicit[:5, -1] = False  # masked although finite: the explicit mask wins
    inf_int = base[3].x_int.clone()
    inf_int[-1, 0] = float("inf")
    times = torch.cumsum(torch.linspace(0.5, 1.5, base[4].length, dtype=torch.float64), 0)
    return [
        dataclasses.replace(base[0], x_obs=nan_obs),  # NaN, default mask
        dataclasses.replace(base[1], obs_mask=explicit, obs_times=times.clone()),
        base[2],  # nothing to store: a null row
        dataclasses.replace(base[3], x_int=inf_int),  # schema 2, but x_obs needs no mask
        dataclasses.replace(base[4], obs_times=times),  # grid only
        base[5],  # a null row of the continuous generator
    ]


def test_schema_2_round_trip_with_nan_null_rows_masks_and_realign(tmp_path):
    eps = _schema_2_episodes()
    meta = _meta()
    suite_dir = _release_io.write_suite(
        meta, eps, tmp_path / "s2", package_version="t", seed=0, shard_size=2
    )
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "2"

    # Every shard has the explicit types and keeps the schema-1 column types,
    # including shard 1, where both new columns are null in every row, and
    # shard 2, where obs_mask is.
    v1_dir = _release_io.write_suite(
        meta, _episodes(n=1), tmp_path / "s1", package_version="t", seed=0
    )
    v1_schema = pq.read_schema(v1_dir / "shard-0000.parquet")
    for shard in manifest["shards"]:
        schema = pq.read_schema(suite_dir / shard["file"])
        assert schema.names == [*_V1_COLUMNS, "obs_times", "obs_mask"]
        assert [schema.field(c).type for c in _V1_COLUMNS] == [
            v1_schema.field(c).type for c in _V1_COLUMNS
        ]
        assert schema.field("obs_times").type == pa.list_(pa.float64())
        assert schema.field("obs_mask").type == pa.list_(pa.bool_())

    got = list(_release_io.read_suite(meta, suite_dir))
    assert len(got) == len(eps)
    for orig, back in zip(eps, got, strict=True):
        assert torch.equal(orig.x_obs.isnan(), back.x_obs.isnan())
        assert torch.equal(orig.x_obs.nan_to_num(), back.x_obs.nan_to_num())
        assert torch.equal(orig.x_int, back.x_int)
        assert torch.equal(orig.y_true, back.y_true)
        assert back.metadata["query_time_idx"] == orig.metadata["query_time_idx"]
    assert torch.equal(got[0].obs_mask, torch.isfinite(eps[0].x_obs))
    assert got[0].obs_mask.dtype == torch.bool
    assert not got[0].obs_mask[3, 1]
    assert int((~got[0].obs_mask).sum()) == 3
    assert torch.equal(got[1].obs_mask, eps[1].obs_mask)
    for i in (1, 4):
        assert got[i].obs_times.dtype == torch.float64
        assert torch.equal(got[i].obs_times, eps[i].obs_times)
    assert [ep.obs_times is None for ep in got] == [True, False, True, True, False, True]
    assert [ep.obs_mask is None for ep in got] == [False, False, True, True, True, True]
    assert torch.isinf(got[3].x_int[-1, 0])

    # Realignment moves each mask column with its x_obs column.
    n = got[1].n_vars
    perm = [*range(1, n), 0]
    fixed = realign_episode(got[1], perm, hidden_canonical=(0,))
    assert torch.equal(fixed.obs_mask, got[1].obs_mask[:, perm])
    assert torch.equal(fixed.x_obs[:, 1:], got[1].x_obs[:, perm[1:]])
    assert not fixed.obs_mask[:5, n - 2].any()  # the masked last column moved left
    assert torch.equal(fixed.obs_times, got[1].obs_times)
    assert realign_episode(got[2], [*range(1, got[2].n_vars), 0]).obs_mask is None


def test_unknown_schema_version_is_refused(tmp_path):
    meta = _meta()
    suite_dir = _release_io.write_suite(
        meta, _episodes(n=1), tmp_path / "s", package_version="t", seed=0
    )
    manifest_path = suite_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "1"
    manifest["schema_version"] = "3"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"schema_version '3'.*'1', '2'"):
        _release_io.read_suite(meta, suite_dir)


def test_numpy_metadata_is_serialized(tmp_path):
    ep = _episodes(n=1)[0]
    ep.metadata.update(
        {
            "f32": np.float32(0.5),
            "f64": np.float64(0.1),
            "i64": np.int64(3),
            "flag": np.bool_(True),
            "arr": np.arange(3),
        }
    )
    meta = _meta()
    suite_dir = _release_io.write_suite(meta, [ep], tmp_path / "s", package_version="t", seed=0)
    (back,) = _release_io.read_suite(meta, suite_dir)
    assert [back.metadata[k] for k in ("f32", "f64", "i64", "flag", "arr")] == [
        0.5,
        0.1,
        3,
        True,
        [0, 1, 2],
    ]


def test_observation_fields_must_match_the_trajectory(tmp_path):
    ep = _episodes(n=1)[0]
    short = dataclasses.replace(ep, obs_times=torch.arange(ep.length - 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="obs_times"):
        _release_io.write_suite(_meta(), [short], tmp_path / "a", package_version="t", seed=0)
    bad_mask = dataclasses.replace(ep, obs_mask=torch.ones(ep.length, ep.n_vars + 1).bool())
    with pytest.raises(ValueError, match="obs_mask"):
        _release_io.write_suite(_meta(), [bad_mask], tmp_path / "b", package_version="t", seed=0)


def test_croissant_lists_observation_fields_only_for_schema_2():
    pytest.importorskip("yaml")
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
    spec = importlib.util.spec_from_file_location("build_release", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def field_ids(version):
        descriptor = mod.croissant_metadata(_meta(), {"schema_version": version, "shards": []})
        assert descriptor["cr:schemaVersion"] == version
        return [f["@id"] for f in descriptor["cr:recordSet"]["field"]]

    v1 = ["x_obs", "x_int", "y_true", "structure", "tier"]
    assert field_ids("1") == v1
    assert field_ids("2") == [*v1, "obs_times", "obs_mask"]
