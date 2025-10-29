from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMClient


logger = logging.getLogger(__name__)

Sentiment = Literal["positif", "neutre", "negatif"]
Impact = Literal["faible", "moyen", "fort"]


class Classification(BaseModel):
    category: str = Field(..., description="Catégorie principale courte et claire")
    subcategories: List[str] = Field(
        ..., min_length=1, description="1 à 3 sous-catégories plus spécifiques"
    )
    keywords: List[str] = Field(
        ..., min_length=1, description="3 à 8 mots-clés courts et pertinents"
    )
    sentiment: Sentiment
    emotional_tone: str = Field(..., description="tonalité émotionnelle dominante")
    summary: str = Field(..., description="phrase synthétique")
    estimated_impact: Impact


SYSTEM_PROMPT = (
    "Tu es un analyste de feedbacks francophone (tous domaines: produit, service, retail, SaaS, logistique, etc.)."
    " Classifie chaque feedback de façon concise et cohérente."
    " Répond STRICTEMENT par un unique objet JSON valide sans texte autour."
    " Les catégories doivent être courtes, singulières, et naturelles dans le contexte du feedback."
    " N'utilise aucune liste prédéfinie: invente la meilleure étiquette courte."
    " Sous-catégories: 1 à 3 éléments plus précis que la catégorie."
    " keywords: 3 à 8 mots-clés en minuscule, sans ponctuation, séparés, pas de doublons (ex: livraison, délai, facture)."
    " sentiment ∈ {positif, neutre, negatif}."
    " estimated_impact ∈ {faible, moyen, fort}."
    " emotional_tone: un seul mot clé (ex: frustration, satisfaction, curiosite, colere, deception, espoir, confusion)."
    " Le résumé est une phrase courte (<= 25 mots)."
)


def build_user_prompt(text: str) -> str:
    schema_hint = {
        "category": "str",
        "subcategories": ["str", "str"],
        "keywords": ["str", "str"],
        "sentiment": "positif|neutre|negatif",
        "emotional_tone": "str",
        "summary": "str",
        "estimated_impact": "faible|moyen|fort",
    }
    return (
        "Feedback:\n" + text.strip() + "\n\n" +
        "Rends un JSON suivant ce schema (valeurs réelles, pas d'exemples):\n" +
        str(schema_hint)
    )


def categorize_text(client: LLMClient, text: str) -> Classification:
    user = build_user_prompt(text)
    # Enforce strict JSON schema so arrays are never empty in model output
    schema = Classification.model_json_schema()
    json_schema = {
        "name": "classification_schema",
        "strict": True,
        "schema": schema,
    }
    data = client.chat_json(
        SYSTEM_PROMPT,
        [{"role": "user", "content": user}],
        json_schema=json_schema,
    )
    try:
        cls = Classification.model_validate(data)
    except ValidationError as e:
        logger.error("Validation error", extra={"errors": e.errors(), "data": data})
        raise
    return cls
