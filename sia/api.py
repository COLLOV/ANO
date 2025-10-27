from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_settings
from .llm_client import LLMClient
from .categorizer import categorize_text, Classification
from .taxonomy import Taxonomy


app = FastAPI(title="SIA Feedback Categorizer")
settings = load_settings()
client = LLMClient(settings)
taxo = Taxonomy()


class CategorizeRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class CategorizeResponseItem(BaseModel):
    category: str
    subcategories: List[str]
    sentiment: str
    emotional_tone: str
    summary: str
    estimated_impact: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": settings.llm_mode,
        "model": client.model,
    }


@app.post("/categorize", response_model=List[CategorizeResponseItem])
def categorize(req: CategorizeRequest):
    inputs: List[str] = []
    if req.text:
        inputs = [req.text]
    elif req.texts:
        inputs = [t for t in req.texts if t]
    else:
        return []

    results: List[CategorizeResponseItem] = []
    for t in inputs:
        cls: Classification = categorize_text(client, t)
        cat, subs = taxo.align(cls.category, cls.subcategories)
        results.append(
            CategorizeResponseItem(
                category=cat,
                subcategories=subs,
                sentiment=cls.sentiment,
                emotional_tone=cls.emotional_tone,
                summary=cls.summary,
                estimated_impact=cls.estimated_impact,
            )
        )
    return results


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008, log_level=settings.log_level.lower())


if __name__ == "__main__":  # pragma: no cover
    main()

