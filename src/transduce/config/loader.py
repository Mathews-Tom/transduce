"""YAML config loader with environment variable substitution.

Implements P1-CFG-01: a single ``load_config(path)`` entry point that
expands ``${VAR}`` and ``${VAR:-default}`` placeholders against the
process environment, parses the YAML body, and validates the result
against the Pydantic schema in :mod:`transduce.config.schema`. Failures
raise :class:`ConfigError` with the file path and the underlying cause
so operators see exactly which key broke startup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from transduce.config.schema import Config

_ENV_VAR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


class ConfigError(RuntimeError):
    """Raised when a configuration file fails to load or validate."""


def load_config(path: Path | str, *, env: dict[str, str] | None = None) -> Config:
    """Load and validate a transduce YAML configuration file.

    Args:
        path: Path to the YAML config file.
        env: Mapping used for ``${VAR}`` substitution. Defaults to ``os.environ``.

    Returns:
        Fully validated :class:`Config` instance.

    Raises:
        ConfigError: file missing, YAML invalid, env reference unresolved,
            or schema validation failed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    environment = env if env is not None else dict(os.environ)
    substituted = _substitute_env_vars(raw_text, environment, source=config_path)

    try:
        parsed = yaml.safe_load(substituted)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"config file {config_path} must contain a YAML mapping at top level")

    try:
        return Config.model_validate(parsed)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}: {exc}") from exc


def _substitute_env_vars(text: str, environment: dict[str, str], *, source: Path) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` placeholders in ``text``."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        raise ConfigError(
            f"unresolved environment variable {name!r} in {source} "
            f"(set the variable or provide a ${{{name}:-default}} fallback)"
        )

    return _ENV_VAR_PATTERN.sub(replace, text)
