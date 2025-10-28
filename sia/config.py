from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_mode: str
    openai_base_url: str
    openai_api_key: str | None
    vllm_base_url: str | None
    model_api: str
    model_local: str
    log_level: str
    cli_workers: int
    api_workers: int
    consolidation_batch_size: int


def load_settings() -> Settings:
    load_dotenv(override=False)

    llm_mode = os.getenv("LLM_MODE", "api").strip().lower()
    if llm_mode not in {"api", "local"}:
        raise ValueError("LLM_MODE must be 'api' or 'local'")

    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    vllm_base_url = os.getenv("VLLM_BASE_URL")

    model_api = os.getenv("LLM_MODEL_API", "gpt-4o-mini").strip()
    # Accept legacy variable name too, without silently hiding errors.
    model_local = os.getenv("LLM_MODEL_LOCAL") or os.getenv("Z_LOCAL_MODEL") or "llama-3.1-8b-instruct"

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Workers: configurables via .env
    try:
        cli_workers = int(os.getenv("CLI_WORKERS", "20"))  # défaut historique du CLI
    except ValueError:
        raise ValueError("CLI_WORKERS must be an integer")

    try:
        api_workers = int(os.getenv("API_WORKERS", "1"))  # uvicorn défaut conservé
    except ValueError:
        raise ValueError("API_WORKERS must be an integer")

    try:
        consolidation_batch_size = int(os.getenv("CONSOLIDATION_BATCH_SIZE", "500"))
    except ValueError:
        raise ValueError("CONSOLIDATION_BATCH_SIZE must be an integer")

    if llm_mode == "api" and not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required in API mode")

    if llm_mode == "local" and not vllm_base_url:
        raise RuntimeError("VLLM_BASE_URL is required in local mode")

    return Settings(
        llm_mode=llm_mode,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        vllm_base_url=vllm_base_url,
        model_api=model_api,
        model_local=model_local,
        log_level=log_level,
        cli_workers=cli_workers,
        api_workers=api_workers,
        consolidation_batch_size=consolidation_batch_size,
    )
