#!/usr/bin/env python3
"""Stage 2: train on pseudo masks with three rounds of self-refinement, then evaluate."""

import os
import sys

from grsr.common import (
    normalize_cli_aliases,
    print_dataset_summary,
    require_data_root,
    resolve_supervision_dir,
)


def main() -> None:
    normalize_cli_aliases(sys.argv)
    if "-h" not in sys.argv and "--help" not in sys.argv:
        data_root = require_data_root(sys.argv)
        print_dataset_summary(data_root, need_splits=True)
        resolve_supervision_dir(data_root, sys.argv)

    from grsr.training.trainer import PointSupTrainer, parse_args

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    PointSupTrainer(args).run()


if __name__ == "__main__":
    main()
