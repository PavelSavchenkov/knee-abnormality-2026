"""Offline mapping of image labels (status, certainty) to a probability of the finding meeting its definition.

The defaults below are a starting point; refit them per finding on the gold studies (share of gold positives in each
status x certainty cell, smoothed towards these values) once the gold runs are back.
"""

DEFAULT = {
    "meets": {"high": 0.95, "moderate": 0.85, "low": 0.7},
    "borderline": {"high": 0.55, "moderate": 0.5, "low": 0.45},
    "possible": {"high": 0.4, "moderate": 0.35, "low": 0.3},
    "below": {"high": 0.1, "moderate": 0.15, "low": 0.2},
    "excluded": {"high": 0.05, "moderate": 0.08, "low": 0.12},
    "normal": {"high": 0.02, "moderate": 0.04, "low": 0.08},
}


def probability(status, certainty, table=DEFAULT):
    """Probability for one finding label; None when the finding was not assessable."""
    if status == "not_assessable":
        return None
    return table[status][certainty]
