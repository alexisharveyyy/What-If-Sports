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
    multitask_path = "models/saved/multitask_transformer_best.pt"

    sim = WhatIfSimulator()
    if os.path.exists(multitask_path):
        try:
            print(f"Loading multi-task transformer from {multitask_path}")
            sim.load_model(multitask_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: Failed to load {multitask_path} ({exc}). "
                "/simulate will return 503 until a compatible checkpoint is saved."
            )
            sim.model = None
    else:
        print(
            "WARNING: No multi-task transformer checkpoint found. "
            "Run `python train/train_multitask_transformer.py` first; "
            "/simulate will return 503 in the meantime."
        )

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
