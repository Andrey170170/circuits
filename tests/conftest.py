"""Shared pytest fixtures for circuit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TEXAS_PICKLE = Path(os.environ.get("CIRCUITS_RESULTS_DIR", "results")) / (
    "case_studies/texas_circuit.pkl"
)


@pytest.fixture(scope="session")
def texas_circuit():
    """Load Texas circuit from pickle.

    Returns None if the pickle file doesn't exist (allows tests to skip gracefully).
    """
    if not TEXAS_PICKLE.exists():
        return None
    # Keep collection of lightweight tracing tests independent of optional
    # analysis/nnsight dependencies.
    from circuits.analysis.circuit_ops import Circuit

    return Circuit.load_from_pickle(str(TEXAS_PICKLE))


@pytest.fixture(scope="session")
def texas_circuit_required(texas_circuit):
    """Same as texas_circuit but skips if not available."""
    if texas_circuit is None:
        pytest.skip(f"Texas circuit pickle not found at {TEXAS_PICKLE}")
    return texas_circuit
