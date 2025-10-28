from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Iterable, Optional

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


def heuristic_category_mapping(
    categories: Iterable[str],
    freq: Dict[str, int],
    *,
    synonyms: Optional[Dict[str, List[str]]] = None,
    max_categories: Optional[int] = None,
    threshold: float = 0.86,
) -> Dict[str, str]:
    """Build a fast heuristic mapping to reduce category variants before LLM.

    - Applies synonym rules (canonical -> [synonyms])
    - Greedy clustering by similarity with difflib
    - Optionally constrains to top-K frequent canonical categories
    """
    import difflib

    cats = list(set(categories))
    if not cats:
        return {}

    # Normalize synonyms dict
    syn_map: Dict[str, str] = {}
    if synonyms:
        for canon, syns in synonyms.items():
            c = canonicalize(canon)
            syn_map[c] = c  # self mapping
            for s in syns:
                syn_map[canonicalize(s)] = c

    # Seed canonical set by frequency (top-K if provided)
    sorted_by_freq = sorted(cats, key=lambda x: (-freq.get(x, 0), x))
    if max_categories and max_categories > 0:
        seeds = sorted_by_freq[: max_categories]
    else:
        seeds = [sorted_by_freq[0]]

    canonical: List[str] = []
    mapping: Dict[str, str] = {}

    # Apply synonyms first
    for cat in cats:
        cn = canonicalize(cat)
        if cn in syn_map:
            mapping[cat] = syn_map[cn]

    # Initialize canonical with seeds
    for s in seeds:
        if s not in mapping:
            canonical.append(s)
            mapping[s] = s

    def best_match(target: str, candidates: List[str]) -> Tuple[str, float]:
        best, score = target, 0.0
        for c in candidates:
            r = difflib.SequenceMatcher(None, target, c).ratio()
            if r > score:
                best, score = c, r
        return best, score

    # Greedy clustering
    for cat in cats:
        if cat in mapping:
            continue
        if canonical:
            bm, sc = best_match(cat, canonical)
            if sc >= threshold:
                mapping[cat] = bm
                continue
        # New canonical if allowed (no max) else map to closest seed anyway
        if not max_categories:
            canonical.append(cat)
            mapping[cat] = cat
        else:
            bm, _ = best_match(cat, canonical)
            mapping[cat] = bm

    return mapping


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
        " Si des mappings existants sont fournis (existing_*_mapping), tu les respectes et tu LES ÉTENDS sans casser les choix déjà faits."
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

    def consolidate_batched(
        self,
        categories: Set[str],
        subcategories: Set[str],
        *,
        batch_size: int = 500,
        on_progress: callable | None = None,
        seed_category_mapping: Optional[Dict[str, str]] = None,
        seed_subcategory_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Consolider en plusieurs appels LLM pour éviter les dépassements.

        La consolidation est progressive: on fournit les mappings existants à chaque lot
        pour préserver la cohérence cross-batch.
        """
        def chunks(lst: List[str], size: int):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        cat_map: Dict[str, str] = dict(seed_category_mapping or {})
        sub_map: Dict[str, str] = dict(seed_subcategory_mapping or {})

        cats = sorted(categories)
        subs = sorted(subcategories)

        # Catégories par lots
        for cat_chunk in chunks(cats, max(1, batch_size)):
            payload = {
                "categories": cat_chunk,
                "subcategories": [],
                "existing_category_mapping": cat_map,
                "existing_subcategory_mapping": sub_map,
                "regles": [
                    "utiliser des libellés courts",
                    "minuscule",
                    "singulier",
                    "fusionner les synonymes évidents",
                    "ne pas inventer de nouvelles catégories hors consolidation",
                ],
            }
            data = self.client.chat_json(self.SYSTEM, [{"role": "user", "content": str(payload)}])
            part_cat = data.get("category_mapping") or {}
            if not isinstance(part_cat, dict):
                raise ValueError("LLM consolidation (cat batch): mapping invalide")
            cat_map.update(part_cat)
            if on_progress:
                on_progress(1)

        # Sous-catégories par lots
        for sub_chunk in chunks(subs, max(1, batch_size)):
            payload = {
                "categories": [],
                "subcategories": sub_chunk,
                "existing_category_mapping": cat_map,
                "existing_subcategory_mapping": sub_map,
                "regles": [
                    "utiliser des libellés courts",
                    "minuscule",
                    "singulier",
                    "fusionner les synonymes évidents",
                    "ne pas inventer de nouvelles catégories hors consolidation",
                ],
            }
            data = self.client.chat_json(self.SYSTEM, [{"role": "user", "content": str(payload)}])
            part_sub = data.get("subcategory_mapping") or {}
            if not isinstance(part_sub, dict):
                raise ValueError("LLM consolidation (sub batch): mapping invalide")
            sub_map.update(part_sub)
            if on_progress:
                on_progress(1)

        return cat_map, sub_map
