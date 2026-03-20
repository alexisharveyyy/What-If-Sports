"""FastAPI backend for What-If Sports NIL Simulator."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import players, simulate
from simulator.comparator import CohortComparator
from simulator.engine import WhatIfSimulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    # Try to load the best saved model
    model_paths = {
        "lstm": "models/saved/lstm_best.pt",
        "transformer": "models/saved/transformer_best.pt",
    }

    sim = WhatIfSimulator()
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            print(f"Loading {model_type} model from {path}")
            sim.model_type = model_type
            sim.load_model(path)
            break
    else:
        print("WARNING: No trained model found. /simulate will return 503.")

    simulate.simulator = sim
    simulate.comparator = CohortComparator()

    yield

    # Cleanup
    simulate.simulator = None
    simulate.comparator = None


app = FastAPI(
    title="What-If Sports NIL Simulator",
    description="Hypothetical NIL valuation forecasting for collegiate athletes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulate.router)
app.include_router(players.router)

# Serve frontend static files
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/health")
async def health():
    has_model = simulate.simulator is not None and simulate.simulator.model is not None
    return {"status": "ok", "model_loaded": has_model}
