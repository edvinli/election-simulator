"""Backward compatibility module for seat uncertainty diagnostics."""

from .diagnostics import attribute_seat_uncertainty, calculate_seat_uncertainty_diagnostics

__all__ = ["calculate_seat_uncertainty_diagnostics", "attribute_seat_uncertainty"]
