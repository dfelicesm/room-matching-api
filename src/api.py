from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import PRECOMPUTED_PATH
from src.load_data import load_internal_precomputed
from src.matching import match_supplier_room
from src.prepare_internal_catalog import build_precomputed_catalog

INTERNAL_DF: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global INTERNAL_DF

    if not Path(PRECOMPUTED_PATH).exists():
        build_precomputed_catalog(PRECOMPUTED_PATH)

    INTERNAL_DF = load_internal_precomputed(PRECOMPUTED_PATH)
    yield
    INTERNAL_DF = None


app = FastAPI(title="Hotel Room Matcher", version="1.0.0", lifespan=lifespan)


@app.post("/match")
def match(data: dict):
    hotel_id = data.get("hotel_id")
    room_name = data.get("room_name")
    if not hotel_id or not room_name:
        raise HTTPException(
            status_code=400, detail="hotel_id and room_name are required"
        )

    scored = match_supplier_room(INTERNAL_DF, hotel_id, room_name)

    cols = [
        "room_name",
        "similarity",
    ]
    resp = scored[cols].copy()

    return {
        "hotel_id": hotel_id,
        "supplier_room_name": room_name,
        "matches": resp.to_dict(orient="records"),
    }
