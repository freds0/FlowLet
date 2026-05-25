"""Validate local FOMO300K T1w archive members before FlowLet training."""

import argparse

from flowlet.data.fomo300k_dataset import FOMO300KDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local FOMO300K T1w samples with age metadata.")
    parser.add_argument("--data-dir", default="/root/DATASETS/FOMO300K_brain_age")
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = FOMO300KDataset(
        data_dir=args.data_dir,
        target_shape=(1, 1, 1),
        normalize=False,
        max_samples=args.max_samples,
        validate_archives=True,
        validation_cache_file=args.cache_file,
        validation_workers=args.workers,
    )
    print(f"Validation complete: {len(dataset)} valid age-labelled T1w scans.")


if __name__ == "__main__":
    main()
