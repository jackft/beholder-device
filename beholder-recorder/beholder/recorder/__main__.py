import logging
import pathlib

import click
from click_loglevel import LogLevel  # type: ignore

from .recorder import Controller
from .utils import setup_logging


@click.group()
def cli():
    pass

@cli.command()
@click.option("--config", "-c", type=click.Path(), required=True, help="config file: beholder.ini")
@click.option("--log-level", "-l", default=logging.INFO, type=LogLevel())
def record(config, log_level):
    setup_logging(pathlib.Path(config), log_level)
    controller = Controller.from_file(pathlib.Path(config))
    controller.run()


if __name__ == "__main__":
    cli()
