"""CLI entry point for c5_SNN pipeline."""

import click


@click.group()
def cli() -> None:
    """c5_SNN: Spiking Neural Network time-series forecasting pipeline."""
