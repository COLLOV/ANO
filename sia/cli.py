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
    parser.add_argument("--workers", type=int, default=20, help="Nombre de workers en parallèle (défaut: 20)")
    parser.add_argument("--consolidate", action="store_true", help="Consolider les catégories/sous-catégories via LLM (2 passes)")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Format de sortie (défaut: jsonl)")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    client = LLMClient(settings)
    taxo = Taxonomy()
    consolidator = LLMTaxonomyConsolidator(client) if args.consolidate else None

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
    for row in read_csv_rows(args.input):
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if rows and args.format == "csv":
        input_fieldnames = list(rows[0].keys())
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

    logging.info("Traitement en parallèle: %d workers", args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_process, rows):
            if res is None:
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

    # Consolidation
    if consolidator and buffered:
        cat_map, sub_map = consolidator.consolidate(all_cats, all_subs)

        # Initialiser csv_writer pour consolidation si nécessaire
        if args.format == "csv" and not csv_writer:
            output_fieldnames = input_fieldnames + [
                "category", "subcategories", "sentiment",
                "emotional_tone", "summary", "estimated_impact"
            ]
            csv_writer = csv.DictWriter(sys.stdout, fieldnames=output_fieldnames)
            csv_writer.writeheader()

        for item in buffered:
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
