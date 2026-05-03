"""Command-line entry point for transduce (P1-CFG-02, P3-BACK-01..09)."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import click
import uvicorn
from opentelemetry import trace

from transduce.api.app import create_app
from transduce.backends.factory import build_backend
from transduce.config.loader import ConfigError, load_config
from transduce.config.schema import BackendEntry, Config
from transduce.observability import build_tracer_provider
from transduce.verification.cosine import CosineSimilarityScorer, build_fastembed_embedder
from transduce.verification.preservation import (
    EntityPreservationScorer,
    NumberPreservationScorer,
    UrlPreservationScorer,
    build_spacy_entity_extractor,
)


def _project_version() -> str:
    try:
        return importlib_metadata.version("transduce")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0"


@click.group()
@click.version_option(version=_project_version(), prog_name="transduce")
def cli() -> None:
    """transduce service CLI."""


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a transduce.yaml configuration file.",
)
@click.option(
    "--host",
    default=None,
    help="Override the host bound from config.service.host.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Override the port bound from config.service.port.",
)
def serve(
    config_path: str, host: str | None, port: int | None
) -> None:  # pragma: no cover — exercised via integration; tested through smoke
    """Start the transduce HTTP service."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    from transduce.verification.base import Scorer

    if config.observability.debug_include_text:
        click.echo(
            "WARN: observability.debug_include_text=true — raw text will be "
            "emitted in OTel spans. Disable in production.",
            err=True,
        )
    provider = build_tracer_provider(config.observability)
    if provider is not None:
        trace.set_tracer_provider(provider)

    scorers: list[Scorer] = [
        CosineSimilarityScorer(
            embed=build_fastembed_embedder(),
            threshold=config.verification.default_cosine_min,
        ),
        EntityPreservationScorer(build_spacy_entity_extractor()),
        NumberPreservationScorer(),
        UrlPreservationScorer(),
    ]
    backend = build_backend(_default_entry(config))
    app = create_app(config, backend=backend, scorers=scorers)
    uvicorn.run(
        app,
        host=host or config.service.host,
        port=port or config.service.port,
    )


def _default_entry(config: Config) -> BackendEntry:
    """Return the BackendEntry referenced by ``config.backends.default``."""
    for entry in config.backends.registry:
        if entry.id == config.backends.default:
            return entry
    raise click.ClickException(  # pragma: no cover — schema validator enforces this
        f"backends.default {config.backends.default!r} missing from registry"
    )


def main() -> None:  # pragma: no cover — entry point
    cli()


if __name__ == "__main__":  # pragma: no cover — module main
    main()


__all__ = ["cli", "main", "serve"]
