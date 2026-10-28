#!/usr/bin/env python3
"""Stage 1: point annotations + infrared images -> continuous initial pseudo masks."""

import sys

from grsr.common import normalize_cli_aliases, print_dataset_summary, require_data_root


def main() -> None:
    normalize_cli_aliases(sys.argv)
    if "-h" not in sys.argv and "--help" not in sys.argv:
        data_root = require_data_root(sys.argv)
        print_dataset_summary(data_root, need_splits=False)

    from grsr.pseudo_labels.generator import main as generate

    generate()


if __name__ == "__main__":
    main()
