from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.utils.logging import get_logger
from nids.utils.paper_export import export_paper_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis-ready CSV tables and publication figures")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root directory to scan for report.json files")
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper_exports",
        help="Directory for aggregated CSV files and publication figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("export_paper_results")
    result = export_paper_results(artifacts_root=args.artifacts_root, output_dir=args.output_dir)
    logger.info("Paper export completed: %s", result)


if __name__ == "__main__":
    main()
