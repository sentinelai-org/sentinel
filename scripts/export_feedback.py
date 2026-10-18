"""Export user feedback into fine-tuning and DPO preference datasets."""

import argparse
import asyncio
import json
from pathlib import Path

from sentinel.core.config import get_settings
from sentinel.evals.feedback import FeedbackStore


async def export_feedback_dataset(output_dir: Path, limit: int = 500) -> None:
    """Extract closed-loop feedback and serialize to training formats."""
    settings = get_settings()
    store = FeedbackStore(redis_url=settings.redis_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_file = output_dir / "fine_tuning_supervised.jsonl"
    dpo_file = output_dir / "dpo_preferences.jsonl"

    print(f"Connecting to Redis at {settings.redis_url}...")
    dataset = await store.export_tuning_dataset(limit=limit)
    stats = await store.get_stats()

    print("\nFeedback Statistics:")
    print(f"  * Total Entries Recorded: {stats.total_feedback}")
    print(f"  * Positive Ratings:        {stats.positive_count}")
    print(f"  * Negative Ratings:        {stats.negative_count}")
    print(f"  * Satisfaction Rate:       {stats.satisfaction_rate}%")

    supervised_count = 0
    dpo_count = 0

    with open(tuning_file, "w", encoding="utf-8") as f_tune, open(
        dpo_file, "w", encoding="utf-8"
    ) as f_dpo:
        for item in dataset:
            if item.get("messages"):
                f_tune.write(json.dumps(item) + "\n")
                supervised_count += 1
            elif item.get("type") == "dpo_negative":
                f_dpo.write(json.dumps(item) + "\n")
                dpo_count += 1

    print("\nExport Completed:")
    print(f"  -> Supervised Fine-Tuning: {supervised_count} samples written to {tuning_file}")
    print(f"  -> DPO Negative Pairs:     {dpo_count} samples written to {dpo_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sentinel feedback for model fine-tuning")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tuning"),
        help="Directory to save exported training datasets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum feedback records to process",
    )
    args = parser.parse_args()
    asyncio.run(export_feedback_dataset(output_dir=args.output_dir, limit=args.limit))


if __name__ == "__main__":
    main()
