"""Command-line entry point for model discovery and evaluation."""
import argparse
import sys

from . import __version__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "metrics":
        from .dlm_metrics.cli import main as metrics_main

        return metrics_main(argv[1:])
    if argv and argv[0] == "train":
        from .cdlm.training import main as train_main

        return train_main(argv[1:])
    parser = argparse.ArgumentParser(prog="open_cdlm", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("models", help="list registered continuous diffusion language models")
    sub.add_parser("metrics", help="evaluation CLI: make-ref or eval")
    sub.add_parser("train", help="train LangFlow or RePlaid on prepared data")
    args = parser.parse_args(argv)
    if args.command == "models":
        from .cdlm import get_model_spec, list_models

        for name in list_models():
            spec = get_model_spec(name)
            print(f"{name}: {', '.join(spec.samplers) or 'custom sampling'}")
    else:
        parser.print_help()

