from __future__ import annotations

import re
import unicodedata
import os
import json
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Dict, List, Set


CANON_SYNONYMS: Dict[str, List[str]] = {
    # Transversal
    "bug": ["bug", "erreur", "incident", "defaut", "plante", "plantage", "crash"],
    "fiabilite": ["fiabilite", "stabilite", "panne", "intermittent", "aléatoire"],
    "performance": ["performance", "lent", "latence", "lag", "ralenti", "degrade"],
    "usabilite": ["usabilite", "ux", "ergonomie", "interface", "ui", "utilisabilite"],
    "acces": ["acces", "accès", "connexion", "login", "auth", "authentification", "identification", "compte", "verrouille"],
    "facturation": ["facturation", "billing", "facture"],
    "prix": ["prix", "tarif", "cout", "coût", "pricing"],
    "paiement": ["paiement", "payment", "cb", "carte", "stripe", "refus"],
    "livraison": ["livraison", "expedition", "expédition", "delai", "retard", "logistique", "tracking"],
    "support": ["support", "service client", "sav", "assistance", "helpdesk"],
    "contenu": ["contenu", "texte", "traduction", "documentation", "doc", "aide", "faq"],
    "fonction manquante": ["fonction manquante", "feature request", "idee", "amélioration", "amelioration", "manque"],
    "securite": ["securite", "sécurité", "confidentialite", "rgpd", "permission", "droits"],
    "qualite": ["qualite", "qualité", "conformite", "defaut produit", "cassé", "casse", "abime"],
    "commande": ["commande", "achat", "panier", "checkout", "validation"],
    "retours": ["retours", "retour", "echange", "échange", "remboursement", "garantie"],
    "compatibilite": ["compatibilite", "compatibilité", "version", "navigateur", "os", "device"],
    "integration": ["integration", "intégration", "api", "webhook", "connecteur"],
    "materiel": ["materiel", "matériel", "imprimante", "scanner", "terminal", "peripherique", "périphérique"],
}

# Optional: extend synonyms via env var TAXONOMY_EXTRA_SYNONYMS (JSON mapping)
_extra = os.getenv("TAXONOMY_EXTRA_SYNONYMS")
if _extra:
    try:
        extra_map = json.loads(_extra)
        if isinstance(extra_map, dict):
            for k, v in extra_map.items():
                if isinstance(v, list) and all(isinstance(s, str) for s in v):
                    CANON_SYNONYMS.setdefault(k, [])
                    CANON_SYNONYMS[k].extend(v)
    except json.JSONDecodeError:
        # Keep strict behavior: invalid JSON should be visible to the operator.
        raise


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
