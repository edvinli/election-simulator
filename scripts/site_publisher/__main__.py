"""CLI for mirroring one certified generation into the website repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.publication_pipeline.pipeline import DEFAULT_PUBLICATION_DIR

from .publisher import SitePublishError, publish_generation_to_site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one certified immutable publication generation into a website "
            "repository. Runs no simulation and makes no Git commit."
        )
    )
    parser.add_argument(
        "--site-repo",
        type=Path,
        required=True,
        help="Path to the website repository checkout (required, never inferred)",
    )
    parser.add_argument(
        "--source-publication-dir",
        type=Path,
        default=DEFAULT_PUBLICATION_DIR,
        help="Certified publication directory in this repository",
    )
    parser.add_argument(
        "--generation",
        default=None,
        help="Generation to mirror; defaults to the one current.json addresses",
    )
    parser.add_argument(
        "--no-pointer",
        action="store_true",
        help="Install the version but do not write the website current.json pointer",
    )
    args = parser.parse_args(argv)
    try:
        report = publish_generation_to_site(
            site_repo=args.site_repo,
            source_publication_dir=args.source_publication_dir,
            generation=args.generation,
            update_pointer=not args.no_pointer,
        )
    except SitePublishError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        "\nNothing was committed or pushed. Review the website working tree, "
        "commit the version directory first, then commit current.json.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
