"""Unit tests for the manifest mode loader (P2-PLG-04)."""

from __future__ import annotations

from pathlib import Path

import pytest

from transduce.registry.manifest import ManifestError, load_manifest

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "manifest_modes"


def test_manifest_only_mode_loads_without_python() -> None:
    spec = load_manifest(_FIXTURE_DIR / "formal-to-warm")

    assert spec.id == "formal-to-warm"
    assert spec.version == "1.0.0"
    assert spec.backend_requirements.min_model_b == pytest.approx(7.0)
    assert spec.verifier_profile.cosine_min == pytest.approx(0.85)
    assert spec.verifier_profile.nli_min == pytest.approx(0.70)
    assert "Rewrite the following text" in spec.prompt_template


def test_manifest_invalid_jinja_raises_at_load_time(tmp_path: Path) -> None:
    package = tmp_path / "broken"
    package.mkdir()
    (package / "mode.toml").write_text(
        "[mode]\n"
        'id = "broken"\n'
        'version = "0.1.0"\n'
        'description = "Broken Jinja."\n'
        'prompt_template = "prompts/broken.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n"
        "\n"
        "[mode.verifier_profile]\n"
        "cosine_min = 0.5\n",
        encoding="utf-8",
    )
    (package / "prompts").mkdir()
    (package / "prompts" / "broken.j2").write_text(
        "{% if invalid syntax %}{% endif %}",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="failed to compile"):
        load_manifest(package)


def test_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="manifest not found"):
        load_manifest(tmp_path / "missing")


def test_manifest_invalid_toml_raises(tmp_path: Path) -> None:
    package = tmp_path / "bad-toml"
    package.mkdir()
    (package / "mode.toml").write_text("not = valid = toml = at all", encoding="utf-8")

    with pytest.raises(ManifestError, match="invalid TOML"):
        load_manifest(package)


def test_manifest_path_traversal_rejected(tmp_path: Path) -> None:
    package = tmp_path / "traversal"
    package.mkdir()
    (package / "mode.toml").write_text(
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "../escape.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "escape.j2").write_text("{{ input }}", encoding="utf-8")

    with pytest.raises(ManifestError, match="must reside under"):
        load_manifest(package)


def test_manifest_unknown_preserve_rule_rejected(tmp_path: Path) -> None:
    package = tmp_path / "unknown-preserve"
    package.mkdir()
    (package / "mode.toml").write_text(
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        'preserve_defaults = ["passwords"]\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
        encoding="utf-8",
    )
    (package / "x.j2").write_text("{{ input }}", encoding="utf-8")

    with pytest.raises(ManifestError, match="unknown preserve rule"):
        load_manifest(package)
