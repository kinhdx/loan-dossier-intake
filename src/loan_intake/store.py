"""SQLite storage for dossier results, reviewer decisions and the model-call journal.

The journal holds token counts, cost and latency per call. It never stores document content or prompts,
so it can be kept and shared for cost reporting without exposing applicant data.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from loan_intake.schemas import Decision, DossierResult, Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS dossiers (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, decision TEXT NOT NULL, status TEXT NOT NULL, result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    dossier_id TEXT NOT NULL, reviewer TEXT NOT NULL, action TEXT NOT NULL, note TEXT, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_calls (
    dossier_id TEXT, step TEXT, model TEXT, input_tokens INT, output_tokens INT, cache_read_tokens INT,
    cache_write_tokens INT, cost_usd REAL, seconds REAL, at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return None
    raise TypeError(type(value))


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def save(self, result: DossierResult) -> None:
        # A dossier missing a required document goes back to the customer, not to the review queue.
        status = {Decision.READY: "ready", Decision.INCOMPLETE: "awaiting_documents"}.get(result.decision, "open")
        payload = json.dumps(dataclasses.asdict(result), default=_json_default, ensure_ascii=False)
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO dossiers VALUES (?,?,?,?,?)",
                             (result.dossier_id, _now(), result.decision.value, status, payload))
        self.journal(result.dossier_id, result.usage)

    def journal(self, dossier_id: str | None, usage: list[Usage]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(dossier_id, u.step, u.model, u.input_tokens, u.output_tokens, u.cache_read_tokens,
                  u.cache_write_tokens, u.cost_usd, u.seconds, _now()) for u in usage])

    def get(self, dossier_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM dossiers WHERE id = ?", (dossier_id,)).fetchone()
        if row is None:
            return None
        reviews = [dict(r) for r in self._db.execute(
            "SELECT reviewer, action, note, at FROM reviews WHERE dossier_id = ? ORDER BY at", (dossier_id,))]
        return {"id": row["id"], "created_at": row["created_at"], "decision": row["decision"],
                "status": row["status"], "result": json.loads(row["result"]), "reviews": reviews}

    def queue(self) -> list[dict]:
        rows = self._db.execute("SELECT id, created_at, decision FROM dossiers WHERE status = 'open' ORDER BY created_at")
        return [dict(r) for r in rows]

    def review(self, dossier_id: str, reviewer: str, action: str, note: str) -> None:
        status = {"accept": "ready", "return": "returned"}[action]
        with self._db:
            self._db.execute("INSERT INTO reviews VALUES (?,?,?,?,?)", (dossier_id, reviewer, action, note, _now()))
            self._db.execute("UPDATE dossiers SET status = ? WHERE id = ?", (status, dossier_id))

    def costs(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT model, step, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
            " ROUND(SUM(cost_usd), 4) AS cost_usd, ROUND(AVG(seconds), 2) AS avg_seconds"
            " FROM model_calls GROUP BY model, step ORDER BY model, step")
        return [dict(r) for r in rows]
