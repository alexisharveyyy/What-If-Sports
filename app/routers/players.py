"""Player data API endpoints."""

import os

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/players", tags=["players"])


def _load_data() -> pd.DataFrame:
    for path in ["data/processed/feature_matrix.csv", "data/sample/sample_players.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@router.get("")
async def list_players(sport: str | None = None, limit: int = 50):
    """List available players."""
    df = _load_data()
    if df.empty:
        return {"players": []}

    latest = df.sort_values("snapshot_week").groupby("player_id").last().reset_index()
    if sport:
        latest = latest[latest["sport"] == sport]

    cols = ["player_id", "sport", "school", "conference", "ppg", "nil_valuation", "nil_tier"]
    available = [c for c in cols if c in latest.columns]
    players = latest[available].head(limit).to_dict("records")
    return {"players": players}


@router.get("/{player_id}/history")
async def player_history(player_id: str):
    """Get full snapshot history for a player."""
    df = _load_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    player_df = df[df["player_id"] == player_id].sort_values("snapshot_week")
    if player_df.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    return {"player_id": player_id, "history": player_df.to_dict("records")}
