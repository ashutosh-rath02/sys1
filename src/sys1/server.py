"""FastAPI server exposing sys1's judgment primitives over HTTP.

    sys1-serve                          # after `pip install -e .[serve]`
    python -m sys1.server               # or directly

Env vars:
    SYS1_ENGINE   "fast" (default, if a fast-encoder checkpoint exists) or "causal"
    SYS1_MODEL    path/name override for the chosen engine
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .engine import DEFAULT_MODEL, Engine
from .fast_engine import FastEngine
from .primitives import Choice, Noul, Score

FAST_ENCODER_DIRS = ["models/sys1-fast-encoder-v2", "models/sys1-fast-encoder"]


def _default_engine():
    engine_type = os.environ.get("SYS1_ENGINE")
    model_override = os.environ.get("SYS1_MODEL")

    if engine_type == "causal":
        return Engine(model_name=model_override or DEFAULT_MODEL)

    fast_dir = model_override or next((d for d in FAST_ENCODER_DIRS if Path(d).is_dir()), None)
    if fast_dir:
        return FastEngine(fast_dir)
    return Engine(model_name=model_override or DEFAULT_MODEL)


app = FastAPI(title="sys1", description="Typed judgments with calibrated probabilities.")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _default_engine()
    return _engine


class ChoiceRequest(BaseModel):
    instructions: str
    options: list[str]
    state: dict | None = None


class NoulRequest(BaseModel):
    instructions: str
    state: dict | None = None


class ScoreRequest(BaseModel):
    instructions: str
    levels: list[str]
    state: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok", "engine": type(get_engine()).__name__}


@app.post("/choice")
def choice(req: ChoiceRequest):
    return Choice(req.options).ask(req.instructions, req.state, get_engine())


@app.post("/noul")
def noul(req: NoulRequest):
    return Noul().ask(req.instructions, req.state, get_engine())


@app.post("/score")
def score(req: ScoreRequest):
    return Score(req.levels).ask(req.instructions, req.state, get_engine())


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
