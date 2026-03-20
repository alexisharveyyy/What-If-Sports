"""Simulation API endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas import SimulationRequest, SimulationResponse

router = APIRouter(prefix="/simulate", tags=["simulate"])

# These will be set by the app lifespan
simulator = None
comparator = None


SUPPORTED_SPORTS = ["basketball", "football"]
CONFERENCES = [
    "SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Independent",
    "AAC", "Mountain West", "Sun Belt", "MAC", "Conference USA",
]


@router.post("", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a what-if NIL simulation."""
    if simulator is None or simulator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        history = [s.model_dump() for s in request.player_history]
        new_snap = request.new_snapshot.model_dump()

        # Single-step simulation
        result = simulator.simulate(history, new_snap)

        # Multi-week timeline
        timeline = simulator.simulate_timeline(
            base_profile=new_snap,
            weeks=request.simulate_weeks_ahead,
        )

        # Cohort comparison
        cohort = comparator.compare(new_snap) if comparator else {}

        return SimulationResponse(
            nil_tier_probs=result["nil_tier_probs"],
            nil_valuation_estimate=result["nil_valuation"],
            direction=result["direction"],
            timeline=timeline,
            cohort_comparison=cohort,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sports")
async def get_sports():
    """Return list of supported sports."""
    return {"sports": SUPPORTED_SPORTS}


@router.get("/conferences")
async def get_conferences():
    """Return list of conferences."""
    return {"conferences": CONFERENCES}
