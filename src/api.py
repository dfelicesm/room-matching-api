from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


app = FastAPI(
    title="Hotel Room Matcher",
    version="1.0.0",
    description="API for matching supplier room names against the internal catalog.",
    lifespan=lifespan,
)


class MatchRequest(BaseModel):
    hotel_id: str = Field(..., example="lp4cd34")
    room_name: str = Field(..., example="Room, 1 King Bed (Hearing Accessible)")


@app.post(
    "/match",
    summary="Find best matches for a supplier room",
    description=(
        "For a given `hotel_id` and `room_name` provided by a supplier, "
        "returns the most similar rooms from the same hotel coming from our internal catalog."
    ),
)
def match(data: MatchRequest):
    """
    Match a supplier's room name to internal catalog entries.
    """
    hotel_id = data.hotel_id
    room_name = data.room_name

    if not hotel_id or not room_name:
        raise HTTPException(
            status_code=400, detail="hotel_id and room_name are required"
        )

    scored = match_supplier_room(INTERNAL_DF, hotel_id, room_name)
    resp = scored[["room_name", "similarity"]].copy()

    return {
        "hotel_id": hotel_id,
        "supplier_room_name": room_name,
        "matches": resp.to_dict(orient="records"),
    }
