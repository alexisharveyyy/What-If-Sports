"""Tests for the FastAPI endpoints."""

import os
import sys

import pytest
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_get_sports(client):
    resp = await client.get("/simulate/sports")
    assert resp.status_code == 200
    data = resp.json()
    assert "sports" in data
    assert "basketball" in data["sports"]


@pytest.mark.anyio
async def test_get_conferences(client):
    resp = await client.get("/simulate/conferences")
    assert resp.status_code == 200
    data = resp.json()
    assert "conferences" in data
    assert len(data["conferences"]) > 0


@pytest.mark.anyio
async def test_list_players(client):
    resp = await client.get("/players")
    assert resp.status_code == 200
    data = resp.json()
    assert "players" in data


@pytest.mark.anyio
async def test_simulate_endpoint(client):
    """Test POST /simulate with a sample payload."""
    payload = {
        "player_history": [
            {
                "sport": "basketball",
                "school": "Duke",
                "conference": "ACC",
                "program_tier": 1,
                "ppg": 18.0,
                "apg": 4.5,
                "rpg": 6.0,
                "spg": 1.2,
                "bpg": 0.8,
                "mpg": 32.0,
                "fg_pct": 0.45,
                "three_pt_pct": 0.38,
                "ft_pct": 0.82,
                "injury_flag": False,
                "games_played": 10,
                "snapshot_week": 1,
            }
        ],
        "new_snapshot": {
            "sport": "basketball",
            "school": "Duke",
            "conference": "ACC",
            "program_tier": 1,
            "ppg": 22.0,
            "apg": 5.0,
            "rpg": 7.0,
            "spg": 1.5,
            "bpg": 1.0,
            "mpg": 34.0,
            "fg_pct": 0.48,
            "three_pt_pct": 0.40,
            "ft_pct": 0.85,
            "injury_flag": False,
            "games_played": 12,
            "snapshot_week": 2,
        },
        "simulate_weeks_ahead": 4,
    }

    resp = await client.post("/simulate", json=payload)

    # Model may not be loaded in test env, so accept 503 or 200
    if resp.status_code == 200:
        data = resp.json()
        assert "nil_tier_probs" in data
        assert "nil_valuation_estimate" in data
        assert "direction" in data
        assert "timeline" in data
        assert "cohort_comparison" in data
        assert len(data["nil_tier_probs"]) == 5
    else:
        assert resp.status_code == 503  # Model not loaded
