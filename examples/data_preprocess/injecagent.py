"""Prepare the bundled InjecAgent splits for the verl CLAP training example.

The corrected source snapshot is checked into ``injecagent_data`` beside this
script, so preprocessing does not need network access. Generated parquet files
are written outside the checkout by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from verl.utils.reward_score.injecagent import (
    INJECAGENT_SOURCE_REPOSITORY,
    INJECAGENT_SOURCE_REVISION,
    build_injecagent_record,
    build_tool_dict,
)

EXPECTED_SPLIT_SIZES = {"train": 310, "eval": 100}
BUNDLED_DATA_DIR = Path(__file__).with_name("injecagent_data")


def prepare_split(data_dir: Path, split: str) -> list[dict]:
    with (data_dir / "raw" / "tools.json").open() as file:
        tool_dict = build_tool_dict(json.load(file))
    with (data_dir / "dataset" / f"{split}.json").open() as file:
        rows = json.load(file)
    expected_size = EXPECTED_SPLIT_SIZES[split]
    if len(rows) != expected_size:
        raise ValueError(f"Expected {expected_size} {split} rows at pinned revision, found {len(rows)}")
    return [build_injecagent_record(row, tool_dict) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional rl-hammer-hardening checkout; defaults to the bundled corrected snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("~/data/injecagent").expanduser())
    args = parser.parse_args()

    data_dir = args.source_dir / "data" / "InjecAgent" if args.source_dir else BUNDLED_DATA_DIR
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_SPLIT_SIZES:
        records = prepare_split(data_dir, split)
        output_path = args.output_dir / f"{split}.parquet"
        Dataset.from_list(records).to_parquet(output_path)
        print(f"Wrote {len(records)} records to {output_path}")

    metadata = {
        "source_repository": INJECAGENT_SOURCE_REPOSITORY,
        "source_revision": INJECAGENT_SOURCE_REVISION,
        "split_sizes": EXPECTED_SPLIT_SIZES,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
