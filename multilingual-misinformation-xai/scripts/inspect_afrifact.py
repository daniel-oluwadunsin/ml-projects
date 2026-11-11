"""Inspect authorized AfrIFact configurations and their real feature schemas."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="masakhane/AfrIFact")
    parser.add_argument("--configuration", default=None)
    args = parser.parse_args()
    try:
        from datasets import get_dataset_config_names, load_dataset_builder
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies with `uv sync` before inspecting dataset metadata"
        ) from exc
    try:
        configurations = get_dataset_config_names(args.repository)
        print(f"Repository: {args.repository}")
        print("Configurations visible to this authenticated session:")
        for config in configurations:
            print(f"  - {config}")
        if args.configuration:
            if args.configuration not in configurations:
                raise SystemExit(f"Configuration {args.configuration!r} is not in that list")
            builder = load_dataset_builder(args.repository, name=args.configuration)
            print(f"Features for {args.configuration!r}: {builder.info.features}")
            print(f"Splits reported in metadata: {builder.info.splits}")
            print("These are source metadata; inspect actual loaded partitions before experiments.")
        else:
            print("Re-run with --configuration NAME to inspect feature names and split metadata.")
    except Exception as exc:
        raise SystemExit(
            f"Dataset metadata inspection failed ({type(exc).__name__}: {exc}). Sign in to Hugging "
            "Face, accept the AfrIFact dataset conditions, then run `hf auth login`. "
            "This tool does "
            "not bypass gated access."
        ) from exc


if __name__ == "__main__":
    main()
