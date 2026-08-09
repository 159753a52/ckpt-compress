"""CLI wrapper for the short residual allocation gate."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.lib.residual_recovery import main  # noqa: E402


if __name__ == "__main__":
    main()
