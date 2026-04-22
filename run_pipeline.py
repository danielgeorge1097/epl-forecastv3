"""
run_pipeline.py — convenience entry point.
Usage: python run_pipeline.py [--backtest] [--no-sim]
Equivalent to: python -m src.pipeline [flags]
"""
import sys
import os

# Ensure src is importable when run from epl-forecasting/
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import main

if __name__ == "__main__":
    main()
