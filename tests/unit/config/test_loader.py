"""Unit tests for the YAML config loader (P1-CFG-01)."""

from __future__ import annotations

from pathlib import Path

import pytest

from transduce.config.loader import ConfigError, load_config

pytestmark = pytest.mark.unit


_MIN_BACKEND = """
backends:
  default: ollama_qwen
  registry:
    - id: ollama_qwen
      provider: ollama
      endpoint: http://localhost:11434
      model: qwen2.5:14b
""".strip()


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "transduce.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_loads_minimal_yaml_with_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MIN_BACKEND))

    assert config.service.host == "0.0.0.0"  # noqa: S104
    assert config.service.port == 8080
    assert config.verification.default_cosine_min == pytest.approx(0.85)
    assert config.language.default == "en"


def test_config_missing_required_field_raises_validation_error(tmp_path: Path) -> None:
    body = "service:\n  port: 9000\n"

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "backends" in str(exc.value).lower()


def test_config_unknown_top_level_section_rejected(tmp_path: Path) -> None:
    body = _MIN_BACKEND + "\nnot_a_section: true\n"

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "Extra" in str(exc.value) or "extra" in str(exc.value)


def test_config_default_backend_must_be_in_registry(tmp_path: Path) -> None:
    body = """
    backends:
      default: missing
      registry:
        - id: ollama_qwen
          provider: ollama
          endpoint: http://localhost:11434
          model: qwen2.5:14b
    """.strip()

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "missing" in str(exc.value)


def test_config_env_substitution_uses_default_when_unset(tmp_path: Path) -> None:
    body = """
    backends:
      default: ollama_qwen
      registry:
        - id: ollama_qwen
          provider: ollama
          endpoint: ${OLLAMA_URL:-http://localhost:11434}
          model: qwen2.5:14b
    """.strip()

    config = load_config(_write(tmp_path, body), env={})

    assert config.backends.registry[0].endpoint == "http://localhost:11434"


def test_config_env_substitution_uses_env_when_set(tmp_path: Path) -> None:
    body = """
    backends:
      default: ollama_qwen
      registry:
        - id: ollama_qwen
          provider: ollama
          endpoint: ${OLLAMA_URL:-http://localhost:11434}
          model: qwen2.5:14b
    """.strip()

    config = load_config(
        _write(tmp_path, body),
        env={"OLLAMA_URL": "http://ollama.internal:11434"},
    )

    assert config.backends.registry[0].endpoint == "http://ollama.internal:11434"


def test_config_unresolved_env_var_without_default_raises(tmp_path: Path) -> None:
    body = """
    backends:
      default: ollama_qwen
      registry:
        - id: ollama_qwen
          provider: ollama
          endpoint: ${MISSING_VAR}
          model: qwen2.5:14b
    """.strip()

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body), env={})

    assert "MISSING_VAR" in str(exc.value)


def test_config_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    body = "service:\n  port: : :\n"

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "invalid YAML" in str(exc.value)


def test_config_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "absent.yaml")

    assert "not found" in str(exc.value)


def test_config_non_mapping_top_level_raises(tmp_path: Path) -> None:
    body = "- a\n- b\n"

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "mapping" in str(exc.value)


def test_config_unknown_provider_in_registry_rejected(tmp_path: Path) -> None:
    body = """
    backends:
      default: anthropic_haiku
      registry:
        - id: anthropic_haiku
          provider: anthropic
          endpoint: https://api.anthropic.com
          model: claude-haiku-4-5
    """.strip()

    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))

    assert "provider" in str(exc.value)


def test_config_loads_example_yaml_succeeds() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    example = repo_root / "transduce.example.yaml"

    config = load_config(example, env={"TRANSDUCE_OLLAMA_ENDPOINT": "http://localhost:11434"})

    assert any(entry.provider == "ollama" for entry in config.backends.registry)


def test_config_modes_default_is_allowlist_with_empty_packages(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MIN_BACKEND))

    assert config.modes.source == "allowlist"
    assert config.modes.packages == []
    assert config.modes.enforce_signing is False


def test_config_modes_allowlist_round_trips_packages(tmp_path: Path) -> None:
    body = (
        _MIN_BACKEND + "\nmodes:\n"
        "  source: allowlist\n"
        "  enforce_signing: false\n"
        "  packages:\n"
        "    - name: transduce-mode-formal-to-warm\n"
        "      version: '1.0.0'\n"
        '      sha256: "' + ("a" * 64) + '"\n'
        "      path: ./packages/formal-to-warm\n"
        '      signed_by: "release@determ-ai"\n'
    )

    config = load_config(_write(tmp_path, body))

    assert len(config.modes.packages) == 1
    entry = config.modes.packages[0]
    assert entry.name == "transduce-mode-formal-to-warm"
    assert entry.signed_by == "release@determ-ai"


def test_config_modes_packages_invalid_sha_length_rejected(tmp_path: Path) -> None:
    body = (
        _MIN_BACKEND + "\nmodes:\n"
        "  packages:\n"
        "    - name: x\n"
        "      version: '1.0.0'\n"
        '      sha256: "deadbeef"\n'
        "      path: ./x\n"
    )

    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_config_modes_source_unknown_rejected(tmp_path: Path) -> None:
    body = _MIN_BACKEND + "\nmodes:\n  source: random\n"

    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))
