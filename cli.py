import json
from typing import Optional
import typer

from src.pipeline import TranslationPipeline
from scripts.benchmark import run_benchmark
from scripts.qc import run_qc_workflow
from scripts.download_models import run_download_all

cli = typer.Typer(help="Multilingual Translation & Quality Control Engine CLI")


@cli.command("translate")
def translate_single(
    text: str = typer.Option(..., "--text", "-tx", help="Source text segment"),
    src: str = typer.Option("en", "--src", "-s", help="Source ISO language code"),
    tgt: str = typer.Option(..., "--tgt", "-t", help="Target ISO language code"),
    dnt: Optional[str] = typer.Option(None, "--dnt", help="Comma-separated DNT terms")
):
    """Translates a single string segment with online masking and QC."""
    dnt_terms = [t.strip() for t in dnt.split(",")] if dnt else None
    pipeline = TranslationPipeline()
    result = pipeline.process_segment(text=text, src_lang=src, tgt_lang=tgt, dnt_terms=dnt_terms)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@cli.command("batch")
def run_batch(
    input_file: str = typer.Option(..., "--input", "-i", help="Input file path (.json/.jsonl)"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output file path"),
    src: str = typer.Option("en", "--src", "-s", help="Source ISO language code"),
    tgt: str = typer.Option(..., "--tgt", "-t", help="Target ISO language code"),
    dnt: Optional[str] = typer.Option(None, "--dnt", help="Comma-separated DNT terms")
):
    """Processes batch input file through translation and quality control."""
    dnt_terms = [t.strip() for t in dnt.split(",")] if dnt else None
    pipeline = TranslationPipeline()
    summary = pipeline.process_file(
        input_file=input_file,
        output_file=output_file,
        src_lang=src,
        tgt_lang=tgt,
        dnt_terms=dnt_terms
    )
    typer.echo(json.dumps(summary, indent=2))


@cli.command("benchmark")
def benchmark_cmd(
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="Custom JSON/JSONL benchmark dataset path"),
    output_file: str = typer.Option("data/outputs/benchmark_results.json", "--output", "-o", help="Output results path")
):
    """Runs comparative benchmarking across Model Family 1 and Model Family 2."""
    run_benchmark(input_path=input_file, output_path=output_file)


@cli.command("qc")
def qc_cmd(
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="Custom JSON QC input dataset"),
    output_file: str = typer.Option("data/outputs/qc_results.json", "--output", "-o", help="Output results path")
):
    """Runs Quality Control validation gates (Entity, FastText LID, Back-Translation)."""
    run_qc_workflow(input_file=input_file, output_file=output_file)


@cli.command("download")
def download_cmd():
    """Pre-downloads and caches all translation and FastText models locally."""
    run_download_all()


if __name__ == "__main__":
    cli()