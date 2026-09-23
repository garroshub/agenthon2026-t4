"""Runtime configuration for the Track 4 submission."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_endpoint: str
    model_id: str
    model_token: str | None
    seed: int
    top_k: int
    timeout_s: float
    max_retries: int
    temperature: float
    max_tokens: int
    unit_timeout_s: float

    @staticmethod
    def from_env() -> "Config":
        return Config(
            model_endpoint=os.environ.get("MODEL_ENDPOINT", "").rstrip("/"),
            model_id=os.environ.get("MODEL_NAME") or os.environ.get("MODEL_ID", ""),
            model_token=os.environ.get("MODEL_TOKEN") or None,
            seed=int(
                os.environ.get("QFBENCH_SEED")
                or os.environ.get("T4_SEED", "20260731")
            ),
            top_k=int(os.environ.get("T4_TOP_K", "10")),
            timeout_s=float(os.environ.get("T4_MODEL_TIMEOUT_S", "60")),
            max_retries=int(os.environ.get("T4_MODEL_RETRIES", "1")),
            temperature=float(os.environ.get("T4_TEMPERATURE", "0")),
            max_tokens=min(4000, int(os.environ.get("T4_MODEL_MAX_TOKENS", "3000"))),
            unit_timeout_s=float(os.environ.get("T4_UNIT_TIMEOUT_S", "540")),
        )
