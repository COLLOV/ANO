from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional

import orjson
from tqdm import tqdm

from .config import load_settings
from .llm_client import LLMClient
from .categorizer import categorize_text
from .taxonomy import Taxonomy, LLMTaxonomyConsolidator


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def read_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def read_xlsx_rows(path: Path) -> Iterable[dict]:
    """Lire les lignes d'un fichier XLSX en dict.

    - Utilise la première feuille
    - La première ligne est l'en-tête
    - Pas de mécanisme de fallback: en cas d'erreur on laisse remonter
    """
    from openpyxl import load_workbook  # dépendance ajoutée via uv

    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return

    headers = [
        (str(h).strip() if h is not None else "")
        for h in header
    ]
    logging.debug("XLSX headers: %s", headers)

    for row in rows_iter:
        # Mappe chaque cellule à son en-tête; si vide, génère un nom de colonne
        out: dict = {}
        for i, val in enumerate(row):
            key = headers[i] if i < len(headers) and headers[i] else f"col_{i+1}"
            out[key] = val
        yield out


def read_rows(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        logging.info("Lecture XLSX: %s", path)
        return read_xlsx_rows(path)
    elif suffix == ".csv" or not suffix:
        # Par défaut on considère CSV si pas d'extension
        logging.info("Lecture CSV: %s", path)
        return read_csv_rows(path)
    else:
        raise ValueError(f"Extension non supportée: {suffix}")


def extract_text(row: dict) -> Optional[str]:
    for key in ("feedback", "text", "message", "comment"):
        if key in row and row[key]:
            return str(row[key])
    # Jira FR style
    if row.get("resume") or row.get("description"):
        return (row.get("resume", "").strip() + ". " + row.get("description", "").strip()).strip()
    return None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Catégorisation de feedbacks via LLM")
    parser.add_argument("--input", type=Path, default=Path("data/tickets_jira.csv"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre de lignes (0 = tout)")
    # Si non fourni, on prendra la valeur depuis .env (CLI_WORKERS)
    parser.add_argument("--workers", type=int, default=None, help="Nombre de workers en parallèle (défaut: CLI_WORKERS ou 20)")
    parser.add_argument("--consolidate", action="store_true", help="Consolider les catégories/sous-catégories via LLM (2 passes)")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Format de sortie (défaut: jsonl)")
    parser.add_argument("--save-intermediate", type=Path, default=None, help="Sauver les résultats intermédiaires (avant consolidation) en JSONL")
    parser.add_argument("--resume-consolidate-from", type=Path, default=None, help="Reprendre la consolidation depuis un JSONL intermédiaire")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    client = LLMClient(settings)
    taxo = Taxonomy()
    consolidator = LLMTaxonomyConsolidator(client) if args.consolidate else None

    if args.resume_consolidate_from and args.limit:
        logging.warning("--limit est ignoré en mode reprise de consolidation")

    if not args.resume_consolidate_from:
        if not args.input.exists():
            logging.error("Input file not found: %s", args.input)
            return 2

    # Préparation de la sortie
    out_f = None
    csv_writer = None
    if args.format == "csv" and args.output:
        out_f = args.output.open("w", newline="", encoding="utf-8")
    elif args.format == "jsonl" and args.output:
        out_f = args.output.open("wb")

    count = 0
    buffered: List[dict] = []
    all_cats: set[str] = set()
    all_subs: set[str] = set()
    input_fieldnames: List[str] = []

    # Pré-lire les lignes (jusqu'à --limit) pour pouvoir paralléliser proprement
    rows: List[dict] = []
    if not args.resume_consolidate_from:
        for row in read_rows(args.input):
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    if rows and args.format == "csv":
        # Construire des en-têtes stables: ordre = 1er row puis ajout des nouvelles colonnes rencontrées
        base = list(rows[0].keys())
        seen = set(base)
        extras_ordered: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    extras_ordered.append(k)
        input_fieldnames = base + extras_ordered
        output_fieldnames = input_fieldnames + [
            "category", "subcategories", "sentiment",
            "emotional_tone", "summary", "estimated_impact"
        ]
        if args.output:
            csv_writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
            csv_writer.writeheader()
        elif not consolidator:
            csv_writer = csv.DictWriter(sys.stdout, fieldnames=output_fieldnames)
            csv_writer.writeheader()

    def _process(row: dict):
        text = extract_text(row)
        if not text:
            return None
        return row, categorize_text(client, text)

    worker_count = args.workers if args.workers is not None else settings.cli_workers
    if not args.resume_consolidate_from:
        logging.info("Traitement en parallèle: %d workers", worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as ex, tqdm(total=len(rows), desc="Traitement", unit="ligne") as pbar:
            for res in ex.map(_process, rows):
                if res is None:
                    pbar.update(1)
                    continue
                row, result = res
                cat, subs = taxo.align(result.category, result.subcategories)

                output_row = row.copy()
                output_row.update({
                    "category": cat,
                    "subcategories": json.dumps(subs) if args.format == "csv" else subs,
                    "sentiment": result.sentiment,
                    "emotional_tone": result.emotional_tone,
                    "summary": result.summary,
                    "estimated_impact": result.estimated_impact,
                })

                if consolidator:
                    buffered.append({
                        "original_row": row,
                        "category": cat,
                        "subcategories": subs,
                        "sentiment": result.sentiment,
                        "emotional_tone": result.emotional_tone,
                        "summary": result.summary,
                        "estimated_impact": result.estimated_impact,
                    })
                    all_cats.add(cat)
                    for s in subs:
                        all_subs.add(s)
                else:
                    if args.format == "csv":
                        if csv_writer:
                            csv_writer.writerow(output_row)
                    else:  # jsonl
                        payload = {
                            "ticket": row.get("ticket_id") or row.get("id") or count + 1,
                            "category": cat,
                            "subcategories": subs,
                            "sentiment": result.sentiment,
                            "emotional_tone": result.emotional_tone,
                            "summary": result.summary,
                            "estimated_impact": result.estimated_impact,
                        }
                        data = orjson.dumps(payload)
                        if out_f:
                            out_f.write(data + b"\n")
                        else:
                            sys.stdout.buffer.write(data + b"\n")

                count += 1
                pbar.update(1)

        # Sauvegarder l'intermédiaire si demandé
        if consolidator and args.save_intermediate:
            logging.info("Sauvegarde de l'intermédiaire: %s", args.save_intermediate)
            with args.save_intermediate.open("wb") as f:
                for item in buffered:
                    f.write(orjson.dumps(item) + b"\n")
    else:
        # Mode reprise: charger le JSONL intermédiaire et configurer consolidation
        if not args.resume_consolidate_from.exists():
            logging.error("Fichier intermédiaire introuvable: %s", args.resume_consolidate_from)
            return 2
        logging.info("Reprise de la consolidation depuis %s", args.resume_consolidate_from)
        consolidator = LLMTaxonomyConsolidator(client)
        buffered = []
        all_cats.clear(); all_subs.clear()
        with args.resume_consolidate_from.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                if not isinstance(obj, dict) or "category" not in obj or "subcategories" not in obj:
                    logging.warning("Entrée intermédiaire invalide ignorée")
                    continue
                buffered.append(obj)
                all_cats.add(obj["category"])
                for s in obj["subcategories"]:
                    all_subs.add(s)

    # Consolidation
    if consolidator and buffered:
        cat_map, sub_map = consolidator.consolidate(all_cats, all_subs)

        # Initialiser csv_writer pour consolidation si nécessaire
        if args.format == "csv" and not csv_writer:
            if not input_fieldnames and buffered:
                # Construire en-têtes depuis original_row (mode reprise)
                base = list((buffered[0].get("original_row") or {}).keys())
                seen = set(base)
                extras: list[str] = []
                for it in buffered:
                    orow = it.get("original_row") or {}
                    for k in orow.keys():
                        if k not in seen:
                            seen.add(k)
                            extras.append(k)
                input_fieldnames = base + extras
            output_fieldnames = input_fieldnames + [
                "category", "subcategories", "sentiment",
                "emotional_tone", "summary", "estimated_impact"
            ]
            csv_writer = csv.DictWriter(sys.stdout, fieldnames=output_fieldnames)
            csv_writer.writeheader()

        for item in tqdm(buffered, desc="Consolidation", unit="ligne"):
            row = item["original_row"]
            cat = cat_map.get(item["category"], item["category"])
            subs = [sub_map.get(s, s) for s in item["subcategories"]]

            if args.format == "csv":
                output_row = row.copy()
                output_row.update({
                    "category": cat,
                    "subcategories": json.dumps(subs),
                    "sentiment": item["sentiment"],
                    "emotional_tone": item["emotional_tone"],
                    "summary": item["summary"],
                    "estimated_impact": item["estimated_impact"],
                })
                if csv_writer:
                    csv_writer.writerow(output_row)
            else:  # jsonl
                payload = {
                    "ticket": row.get("ticket_id") or row.get("id") or "N/A",
                    "category": cat,
                    "subcategories": subs,
                    "sentiment": item["sentiment"],
                    "emotional_tone": item["emotional_tone"],
                    "summary": item["summary"],
                    "estimated_impact": item["estimated_impact"],
                }
                data = orjson.dumps(payload)
                if out_f:
                    out_f.write(data + b"\n")
                else:
                    sys.stdout.buffer.write(data + b"\n")

    logging.info(
        "Processed %d feedback(s). Categories=%d, Subcategories(total)=%d",
        count,
        len(taxo.categories),
        sum(len(v) for v in taxo.subs_by_cat.values()),
    )

    if out_f:
        out_f.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
