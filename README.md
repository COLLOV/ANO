**SIA – Service de catégorisation de feedbacks (API / vLLM local, multi‑domaine)**

- Modes LLM: `api` (fournisseur externe compatible OpenAI) ou `local` (vLLM OpenAI-compatible).
- Charge les variables depuis `.env`.
- Expose un CLI (`sia-categorize`) et une API HTTP (`sia-api`).

Env requis (.env):
- `LLM_MODE` = `api` ou `local`
- `OPENAI_BASE_URL` (mode api, défaut: https://api.openai.com/v1)
- `OPENAI_API_KEY` (mode api)
- `VLLM_BASE_URL` (mode local, ex: http://127.0.0.1:8000/v1)
- `LLM_MODEL_API` (ex: gpt-4o-mini), `LLM_MODEL_LOCAL` (nom du modèle servi par vLLM)

Installation (uv):
1) `uv sync`
2) CLI: `uv run sia-categorize --input data/tickets_jira.csv --limit 5 --workers 20`
3) API: `API_WORKERS=20 uv run sia-api` (puis POST `/categorize`)

CLI:
- Entrée: CSV ou XLSX (première feuille, première ligne = en‑têtes). Colonnes supportées: `feedback`, `text`, `message`, `comment`, ou `resume`+`description` (Jira FR).
- Sort sur stdout des lignes JSON (une par ticket) ou écrit un fichier via `--output`.
 - Option `--consolidate`: 2 passes avec consolidation LLM des libellés (catégories/sous‑catégories) pour éviter les doublons/synonymes.
 - Option `--workers`: parallélise les appels LLM (défaut: 20). Exemple: `--workers 20`.

Exemples CLI:
- CSV → JSONL: `uv run sia-categorize --input data/tickets_jira.csv --limit 5 --format jsonl`
- XLSX → CSV: `uv run sia-categorize --input ./tickets.xlsx --format csv --output ./out.csv`

API:
- `GET /health` → statut (mode, modèle)
- `POST /categorize` body: `{ "text": "..." }` ou `{ "texts": ["..."] }` et option `?consolidate=true` pour regrouper les libellés retournés par le LLM.
 - Démarrage multi‑workers: définir `API_WORKERS=<N>` (ex: `API_WORKERS=20 uv run sia-api`). Pour une mise en prod avancée, vous pouvez aussi utiliser gunicorn: `gunicorn -w 20 -k uvicorn.workers.UvicornWorker sia.api:app`.

Résultat (schéma synthétique):
```
{
  "category": str,
  "subcategories": [str, ...],
  "sentiment": "positif|neutre|negatif",
  "emotional_tone": "frustration|satisfaction|curiosite|...",
  "summary": str,
  "estimated_impact": "faible|moyen|fort"
}
```

Notes de conception:
- Aucune relance silencieuse: si le JSON du LLM est invalide → erreur explicite.
- Pas de taxonomie imposée: le LLM propose les catégories. Une consolidation optionnelle par LLM permet de fusionner les variantes/synonymes.
- Journalisation sobre et utile.

Catégorisation multi‑domaine
- Le prompt et la taxonomie sont génériques (produit, service, retail, SaaS, logistique…).
- Catégories suggérées (non exclusives): bug, fiabilite, performance, usabilite, acces, facturation, prix, paiement, livraison, support, contenu/documentation, fonction manquante, securite, qualite, commande, retours, compatibilite, integration, materiel.
- Extensibilité: ajoutez des synonymes via l’environnement, par ex. `TAXONOMY_EXTRA_SYNONYMS='{"marketing": ["campagne", "promotion"], "partenariat": ["partner", "partenaire"]}'`.

Mode local (vLLM):
- Démarrer un serveur OpenAI-compatible vLLM, par ex.:
  - `python -m vllm.entrypoints.openai.api_server --model <NOM_MODELE> --port 8001`
- `.env`:
  - `LLM_MODE=local`
  - `VLLM_BASE_URL=http://127.0.0.1:8001/v1`
  - `LLM_MODEL_LOCAL=<NOM_MODELE>`
