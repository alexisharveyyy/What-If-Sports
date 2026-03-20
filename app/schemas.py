"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class PlayerSnapshot(BaseModel):
    sport: str = "basketball"
    school: str = "Unknown"
    conference: str = "Unknown"
    program_tier: int = Field(default=3, ge=1, le=4)
    ppg: float = Field(default=0.0, ge=0)
    apg: float = Field(default=0.0, ge=0)
    rpg: float = Field(default=0.0, ge=0)
    injury_flag: bool = False
    games_played: int = Field(default=1, ge=0)
    snapshot_week: int = Field(default=1, ge=1)


class SimulationRequest(BaseModel):
    player_history: list[PlayerSnapshot] = []
    new_snapshot: PlayerSnapshot
    simulate_weeks_ahead: int = Field(default=4, ge=1, le=12)


class SimulationResponse(BaseModel):
    nil_tier_probs: list[float]
    nil_valuation_estimate: float
    direction: str
    timeline: list[dict]
    cohort_comparison: dict
