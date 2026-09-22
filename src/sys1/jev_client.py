"""Thin client for TypeSafe's real Jev API, used only to run head-to-head
comparisons against our own sys1 engine (e.g. eval/snake_vs_jev.py). Not
used anywhere in the sys1 inference path itself.

Reads TYPESAFE_API_KEY from the environment (or a local .env file — never
committed, see .gitignore).
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

API_URL = "https://api.typesafe.ai/v1/systemone"


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.is_file() or os.environ.get("TYPESAFE_API_KEY"):
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv_if_present()


class JevChoiceClient:
    """Mirrors sys1's Choice primitive, but backed by the real Jev API."""

    def __init__(self, api_key: str | None = None, model: str = "jev-latest"):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY not set (env var or .env file)")
        self.model = model

    def ask(self, instructions: str, options: list[str], state: dict | None = None) -> dict:
        body = {
            "state": state or {},
            "model": self.model,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {opt: opt for opt in options},
                }
            },
        }
        resp = requests.post(
            API_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["answers"]["q"]
        return {"choice": answer["choice"], "probabilities": answer["probabilities"]}
