"""Comparative, provenance-first benchmark for ElectionSimulator and Botten Ada."""

from .config import BOTTEN_ADA_SOURCE, PIVOT_RULE, PARTY_ORDER
from .harness import run_benchmark

__all__ = ["BOTTEN_ADA_SOURCE", "PARTY_ORDER", "PIVOT_RULE", "run_benchmark"]
