"""CLI for aidetect."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from aidetect import __version__
from aidetect.detector import Detector


console = Console()


@click.group()
@click.version_option(__version__)
def main():
    """Fine-grained AI text detection with model attribution."""
    pass


@main.command()
@click.argument("input", required=True)
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show feature details")
def analyze(input: str, json_output: bool, verbose: bool):
    """Analyze text or file for AI-generated content."""
    text = _read_input(input)
    if not text:
        console.print("[red]No text to analyze[/red]")
        sys.exit(1)

    detector = Detector()
    result = detector.analyze(text)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _print_result(result, verbose)


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def batch(directory: str, fmt: str, output: str):
    """Batch analyze files in a directory."""
    detector = Detector()
    results = []

    dir_path = Path(directory)
    files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

    if not files:
        console.print("[yellow]No .txt or .md files found[/yellow]")
        return

    for f in sorted(files):
        text = f.read_text(errors="ignore")
        r = detector.analyze(text)
        results.append({
            "file": str(f.name),
            "aggregate_score": r.aggregate_score,
            "dominant_model": r.dominant_model,
            "sentences": len(r.sentences),
            "model_distribution": r.model_distribution,
        })

    if fmt == "json":
        out = json.dumps(results, indent=2)
    else:
        lines = ["file,aggregate_score,dominant_model,sentences"]
        for r in results:
            lines.append(f"{r['file']},{r['aggregate_score']},{r['dominant_model']},{r['sentences']}")
        out = "\n".join(lines)

    if output:
        Path(output).write_text(out)
        console.print(f"[green]Results written to {output}[/green]")
    else:
        click.echo(out)


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(host: str, port: int):
    """Start MCP server for AI text detection."""
    try:
        from aidetect.server import create_app
        import uvicorn
    except ImportError:
        console.print("[red]Install MCP dependencies: pip install aidetect[mcp][/red]")
        sys.exit(1)

    app = create_app()
    console.print(f"[green]Starting MCP server on {host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port)


def _read_input(input_str: str) -> str:
    p = Path(input_str)
    if p.exists() and p.is_file():
        return p.read_text(errors="ignore")
    return input_str


def _print_result(result, verbose: bool = False):
    # Header
    score_color = "red" if result.aggregate_score > 0.7 else "yellow" if result.aggregate_score > 0.4 else "green"
    console.print(f"\n[bold]AI Detection Score:[/bold] [{score_color}]{result.aggregate_score:.1%}[/{score_color}]")
    if result.dominant_model:
        console.print(f"[bold]Dominant Model:[/bold] {result.dominant_model}")
    console.print(f"[bold]Sentences:[/bold] {len(result.sentences)}\n")

    # Sentence table
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Sentence", max_width=60)
    table.add_column("Label", width=10)
    table.add_column("Conf", width=6)

    for i, s in enumerate(result.sentences, 1):
        label_color = "green" if s.label == "human" else "red"
        table.add_row(
            str(i),
            s.text[:80] + ("..." if len(s.text) > 80 else ""),
            f"[{label_color}]{s.label}[/{label_color}]",
            f"{s.confidence:.0%}",
        )

    console.print(table)

    # Model distribution
    if result.model_distribution:
        console.print("\n[bold]Model Distribution:[/bold]")
        for model, pct in sorted(result.model_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct * 20)
            console.print(f"  {model:>8}: {bar} {pct:.0%}")
    console.print()
