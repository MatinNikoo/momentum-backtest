"""trendlab: a small trend following backtester with cash at the T-bill rate and no lookahead."""

from . import engine, rules, metrics, benchmarks, stats, insurance

__all__ = ["engine", "rules", "metrics", "benchmarks", "stats", "insurance"]
__version__ = "0.3.0"
