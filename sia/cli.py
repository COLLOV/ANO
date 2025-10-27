from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import orjson

from .config import load_settings
from .llm_client import LLMClient
from .categorizer import categorize_text
from .taxonomy import Taxonomy


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
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    client = LLMClient(settings)
    taxo = Taxonomy()

    out_f = args.output.open("wb") if args.output else None
    count = 0

    if not args.input.exists():
        logging.error("Input file not found: %s", args.input)
        return 2

    for row in read_csv_rows(args.input):
        text = extract_text(row)
        if not text:
            continue
        result = categorize_text(client, text)
        cat, subs = taxo.align(result.category, result.subcategories)
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
        if args.limit and count >= args.limit:
            break

    logging.info("Processed %d feedback(s). Categories=%d, Subcategories(total)=%d", count, len(taxo.categories), sum(len(v) for v in taxo.subs_by_cat.values()))

    if out_f:
        out_f.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

