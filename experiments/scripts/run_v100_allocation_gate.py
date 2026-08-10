"""CLI wrapper for the short residual allocation gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_short_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
