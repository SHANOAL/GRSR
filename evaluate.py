#!/usr/bin/env python3
"""Evaluate a trained checkpoint (IoU, Pd, Fa) on the dense test masks."""

import sys

from grsr.common import normalize_cli_aliases
from grsr.training.evaluate import main

if __name__ == "__main__":
    normalize_cli_aliases(sys.argv)
    main()
