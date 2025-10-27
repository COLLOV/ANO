from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .llm_client import LLMClient


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-_/]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(term: str) -> str:
    # Minimal normalization only; keep taxonomy LLM-driven.
    return normalize(term)


@dataclass
class Taxonomy:
    categories: Set[str] = field(default_factory=set)
    subs_by_cat: Dict[str, Set[str]] = field(default_factory=dict)

    def align(self, category: str, subcategories: List[str]) -> tuple[str, List[str]]:
        canon_cat = canonicalize(category)
        if canon_cat not in self.categories:
            self.categories.add(canon_cat)
            self.subs_by_cat.setdefault(canon_cat, set())

        aligned_subs: List[str] = []
        for s in subcategories:
            canon_sub = canonicalize(s)
            if canon_sub not in self.subs_by_cat[canon_cat]:
                self.subs_by_cat[canon_cat].add(canon_sub)
            aligned_subs.append(canon_sub)
        return canon_cat, aligned_subs


class LLMTaxonomyConsolidator:
    """LLM-driven consolidation of labels into canonical forms.

    Produces two mappings: category_mapping and subcategory_mapping.
    """

    SYSTEM = (
        "Tu consolides des étiquettes de catégories et sous-catégories issues de feedbacks."
        " Objectif: regrouper les synonymes/variantes vers un libellé canonique court,"
        " en minuscule, singulier si possible. Pas de texte hors JSON."
        " Réponds avec un objet JSON: {\"category_mapping\": {...}, \"subcategory_mapping\": {...}}."
        " Les clés sont les libellés d'entrée EXACTS, les valeurs sont les versions canoniques."
    )

    def __init__(self, client: LLMClient):
        self.client = client

    def consolidate(self, categories: Set[str], subcategories: Set[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        payload = {
            "categories": sorted(categories),
            "subcategories": sorted(subcategories),
            "regles": [
                "utiliser des libellés courts",
                "minuscule",
                "singulier",
                "fusionner les synonymes évidents",
                "ne pas inventer de nouvelles catégories hors consolidation",
            ],
        }
        data = self.client.chat_json(self.SYSTEM, [{"role": "user", "content": str(payload)}])
        cat_map = data.get("category_mapping") or {}
        sub_map = data.get("subcategory_mapping") or {}
        if not isinstance(cat_map, dict) or not isinstance(sub_map, dict):
            raise ValueError("LLM consolidation: mappings manquants ou invalides")
        return cat_map, sub_map
