from __future__ import annotations

import logging
from typing import List, Optional
import os

from fastapi import FastAPI, Query
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
    keywords: List[str]
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
def categorize(req: CategorizeRequest, consolidate: bool = Query(default=False)):
    inputs: List[str] = []
    if req.text:
        inputs = [req.text]
    elif req.texts:
        inputs = [t for t in req.texts if t]
    else:
        return []

    items: List[CategorizeResponseItem] = []
    for t in inputs:
        cls: Classification = categorize_text(client, t)
        cat, subs = taxo.align(cls.category, cls.subcategories)
        items.append(
            CategorizeResponseItem(
                category=cat,
                subcategories=subs,
                keywords=cls.keywords,
                sentiment=cls.sentiment,
                emotional_tone=cls.emotional_tone,
                summary=cls.summary,
                estimated_impact=cls.estimated_impact,
            )
        )

    if consolidate and len(items) > 0:
        from .taxonomy import LLMTaxonomyConsolidator

        cats = {it.category for it in items}
        subs = {s for it in items for s in it.subcategories}
        cat_map, sub_map = LLMTaxonomyConsolidator(client).consolidate(cats, subs)
        for it in items:
            it.category = cat_map.get(it.category, it.category)
            it.subcategories = [sub_map.get(s, s) for s in it.subcategories]
    return items


def main() -> None:  # pragma: no cover
    import uvicorn
    # Nombre de workers configurable via .env (API_WORKERS), centralisé dans Settings
    workers = settings.api_workers
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level=settings.log_level.lower(),
        workers=workers,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
