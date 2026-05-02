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


def _write_minimal_manifest(package: Path, mode_section: str) -> None:
    package.mkdir()
    (package / "x.j2").write_text("{{ input }}", encoding="utf-8")
    (package / "mode.toml").write_text(mode_section, encoding="utf-8")


def test_manifest_missing_mode_section_rejected(tmp_path: Path) -> None:
    package = tmp_path / "no-mode"
    _write_minimal_manifest(package, "[other]\nfoo = 1\n")

    with pytest.raises(ManifestError, match=r"\[mode\] section"):
        load_manifest(package)


def test_manifest_missing_prompt_template_rejected(tmp_path: Path) -> None:
    package = tmp_path / "no-template"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="prompt_template must be a non-empty string"):
        load_manifest(package)


def test_manifest_missing_template_file_rejected(tmp_path: Path) -> None:
    package = tmp_path / "no-template-file"
    package.mkdir()
    (package / "mode.toml").write_text(
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "missing.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="prompt template not found"):
        load_manifest(package)


def test_manifest_backend_requirements_must_be_table(tmp_path: Path) -> None:
    package = tmp_path / "br-not-table"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        'backend_requirements = "oops"\n',
    )

    with pytest.raises(ManifestError, match=r"\[mode\.backend_requirements\] must be a table"):
        load_manifest(package)


def test_manifest_min_model_b_required(tmp_path: Path) -> None:
    package = tmp_path / "no-min-model"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "\n"
        "[mode.backend_requirements]\n",
    )

    with pytest.raises(ManifestError, match="min_model_b is required"):
        load_manifest(package)


def test_manifest_verifier_profile_must_be_table(tmp_path: Path) -> None:
    package = tmp_path / "vp-not-table"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        'verifier_profile = "oops"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match=r"\[mode\.verifier_profile\] must be a table"):
        load_manifest(package)


def test_manifest_intensity_range_must_be_pair(tmp_path: Path) -> None:
    package = tmp_path / "bad-range"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "intensity_range = [0.0, 0.5, 1.0]\n"
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="intensity_range must be"):
        load_manifest(package)


def test_manifest_supported_languages_must_be_list_of_strings(tmp_path: Path) -> None:
    package = tmp_path / "bad-langs"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "supported_languages = [1, 2]\n"
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="supported_languages must be a list of strings"):
        load_manifest(package)


def test_manifest_preserve_defaults_must_be_list(tmp_path: Path) -> None:
    package = tmp_path / "bad-preserve"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        'preserve_defaults = "entities"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="preserve_defaults must be a list"):
        load_manifest(package)


def test_manifest_preserve_defaults_entries_must_be_strings(tmp_path: Path) -> None:
    package = tmp_path / "bad-preserve-entry"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "preserve_defaults = [1, 2]\n"
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="preserve_defaults entries must be strings"):
        load_manifest(package)


def test_manifest_verifier_profile_invalid_threshold_rejected(tmp_path: Path) -> None:
    package = tmp_path / "bad-vp"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = "x"\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n"
        "\n"
        "[mode.verifier_profile]\n"
        "cosine_min = 1.5\n",
    )

    with pytest.raises(ManifestError, match="verifier_profile.*failed validation"):
        load_manifest(package)


def test_manifest_load_manifests_from_directory_skips_non_manifest_dirs(
    tmp_path: Path,
) -> None:
    """``load_manifests_from_directory`` only loads subdirectories with mode.toml."""
    root = tmp_path / "modes"
    root.mkdir()
    valid = root / "ok"
    _write_minimal_manifest(
        valid,
        "[mode]\n"
        'id = "ok"\n'
        'version = "0.1.0"\n'
        'description = "ok"\n'
        'prompt_template = "x.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )
    (root / "not-a-mode").mkdir()
    (root / "stray-file.txt").write_text("noise", encoding="utf-8")

    from transduce.registry.manifest import load_manifests_from_directory

    specs = load_manifests_from_directory(root)

    assert len(specs) == 1
    assert specs[0].id == "ok"


def test_manifest_load_manifests_from_directory_missing_root_raises(tmp_path: Path) -> None:
    from transduce.registry.manifest import load_manifests_from_directory

    with pytest.raises(ManifestError, match="manifest root not found"):
        load_manifests_from_directory(tmp_path / "missing")


def test_manifest_invalid_id_fails_schema_validation(tmp_path: Path) -> None:
    package = tmp_path / "empty-id"
    _write_minimal_manifest(
        package,
        "[mode]\n"
        'id = ""\n'
        'version = "0.1.0"\n'
        'description = "x"\n'
        'prompt_template = "x.j2"\n'
        "\n"
        "[mode.backend_requirements]\n"
        "min_model_b = 1.0\n",
    )

    with pytest.raises(ManifestError, match="failed schema validation"):
        load_manifest(package)
