"""Released suites name the exact bytes of the config that built them.

``scripts/build_release.py`` writes ``config_hash`` into every manifest: the
first 16 hex digits of the SHA-256 of the config file's text, comments
included. These configs still hold the bytes their released versions were built
from, so anyone can match a downloaded manifest to the file in this repository.
Their header comments therefore still describe the suites as prepared and must
stay as they are; a new build goes into a new config or a new version.

The v1.0.0 suites (hash ``237353e82091591e``) were built from
``scripts/release_config.yaml`` as of commit 8e95019. That file changed later
without changing what it generates, which the frozen fingerprints check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# config file -> config_hash in the released manifest (Zenodo record).
RELEASED = {
    "release_config_v1_1.yaml": "bc4c6a32892fcf26",  # dot-Identifiability-v1 1.1.0
    "release_config_v1_2.yaml": "d9b70dba4efc774a",  # dot-Identifiability-v1 1.2.0, 23095083
    "release_config_seasonal_trend.yaml": "9a1dc5a2a2de7066",  # dot-SeasonalTrend-v1, 23095134
    "release_config_wide.yaml": "9caa04a69e4c1f3b",  # dot-Wide-v1, 23095148
    "release_config_observed_v1.yaml": "f497c6cfe3d0ce10",  # dot-Observed-v1, 23095117
    "release_config_continuous_irregular.yaml": "b9cbbb1ef0b8d1dc",  # 23095097
}


@pytest.mark.parametrize(("config", "released_hash"), sorted(RELEASED.items()))
def test_released_config_is_unchanged(config: str, released_hash: str) -> None:
    """A released config hashes to the ``config_hash`` of its released manifest.

    Args:
        config: File name under ``scripts/``.
        released_hash: ``config_hash`` of the released manifest.
    """
    # build_release.py hashes Path.read_text(), so the test does the same.
    text = (_SCRIPTS / config).read_text()
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == released_hash, (
        f"{config} differs from the bytes its released suite was built from"
    )
