import click

from tokenflow._version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="tokenflow")
def main() -> None:
    """Lightweight benchmarking tool for OpenAI-compatible LLM endpoints."""
