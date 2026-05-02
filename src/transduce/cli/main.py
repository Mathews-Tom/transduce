"""Command-line entry point for transduce (P1-CFG-02)."""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import click
import uvicorn

from transduce.api.app import create_app
from transduce.config.loader import ConfigError, load_config
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

    scorers: list[Scorer] = [
        CosineSimilarityScorer(
            embed=build_fastembed_embedder(),
            threshold=config.verification.default_cosine_min,
        ),
        EntityPreservationScorer(build_spacy_entity_extractor()),
        NumberPreservationScorer(),
        UrlPreservationScorer(),
    ]
    app = create_app(config, scorers=scorers)
    uvicorn.run(
        app,
        host=host or config.service.host,
        port=port or config.service.port,
    )


def main() -> None:  # pragma: no cover — entry point
    cli()


if __name__ == "__main__":  # pragma: no cover — module main
    main()


__all__ = ["cli", "main", "serve"]
