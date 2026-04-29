"""Config modules for synthetic NIL dataset generation."""

from pipeline.config.conferences import CONFERENCES, CONFERENCE_WEIGHTS
from pipeline.config.schools import SCHOOLS_BY_CONFERENCE, BLUE_BLOODS, MARKET_SIZE
from pipeline.config.stat_distributions import (
    POSITION_STAT_DISTRIBUTIONS,
    PROGRAM_TIER_PROBS,
    POSITION_PROBS,
    CLASS_YEAR_PROBS,
)

__all__ = [
    "CONFERENCES",
    "CONFERENCE_WEIGHTS",
    "SCHOOLS_BY_CONFERENCE",
    "BLUE_BLOODS",
    "MARKET_SIZE",
    "POSITION_STAT_DISTRIBUTIONS",
    "PROGRAM_TIER_PROBS",
    "POSITION_PROBS",
    "CLASS_YEAR_PROBS",
]
