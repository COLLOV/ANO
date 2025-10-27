from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Dict, List, Set


CANON_SYNONYMS: Dict[str, List[str]] = {
    "bug": ["bug", "erreur", "plantage", "crash"],
    "performance": ["performance", "lent", "latence", "lag"],
    "connexion": ["connexion", "login", "auth", "authentification", "identification", "signin"],
    "impression": ["impression", "printer", "imprimante", "recto-verso"],
    "reseau": ["reseau", "réseau", "vpn", "wifi", "lan"],
    "securite": ["securite", "sécurité", "droits", "permission", "acces", "accès"],
    "facturation": ["facturation", "billing", "paiement", "payment", "facture"],
    "usabilite": ["usabilite", "ux", "ergonomie", "interface", "ui", "utilisabilite"],
    "fonction manquante": ["fonction manquante", "feature request", "idee", "amelioration", "manque"]
}


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-_/]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(term: str, existing: Set[str] | None = None) -> str:
    n = normalize(term)
    # Map through synonym table
    for canon, syns in CANON_SYNONYMS.items():
        if n in map(normalize, syns):
            return canon
    # Fuzzy match existing canon labels if available
    if existing:
        match = get_close_matches(n, list(existing), n=1, cutoff=0.88)
        if match:
            return match[0]
    return n


@dataclass
class Taxonomy:
    categories: Set[str] = field(default_factory=set)
    subs_by_cat: Dict[str, Set[str]] = field(default_factory=dict)

    def align(self, category: str, subcategories: List[str]) -> tuple[str, List[str]]:
        canon_cat = canonicalize(category, self.categories)
        if canon_cat not in self.categories:
            self.categories.add(canon_cat)
            self.subs_by_cat.setdefault(canon_cat, set())

        aligned_subs: List[str] = []
        for s in subcategories:
            canon_sub = canonicalize(s, self.subs_by_cat[canon_cat])
            if canon_sub not in self.subs_by_cat[canon_cat]:
                self.subs_by_cat[canon_cat].add(canon_sub)
            aligned_subs.append(canon_sub)
        return canon_cat, aligned_subs

