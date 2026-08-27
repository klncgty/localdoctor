"""SQLite schema and writes. Active in phase 1; phase 2 only adds a read surface.

Every request is stored — even when no diagnosis is produced. This is where the
answer to "why didn't it warn me?" lives.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from localdoctor.models import Diagnosis, RequestRecord

DEFAULT_DB = Path.home() / ".localdoctor" / "localdoctor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id              TEXT PRIMARY KEY,
  ts              TEXT NOT NULL,
  endpoint        TEXT NOT NULL,
  model           TEXT,
  request_body    BLOB NOT NULL,
  request_headers TEXT NOT NULL,
  response_body   BLOB,
  status_code     INTEGER,
  stream          INTEGER NOT NULL,
  prompt_tokens   INTEGER,
  completion_tokens INTEGER,
  finish_reason   TEXT,
  num_ctx         INTEGER,
  num_ctx_source  TEXT,
  ttft_ms         INTEGER,
  total_ms        INTEGER,
  chunk_count     INTEGER
);

CREATE TABLE IF NOT EXISTS diagnoses (
  id          INTEGER PRIMARY KEY,
  request_id  TEXT NOT NULL REFERENCES requests(id),
  rule_id     TEXT NOT NULL,
  confidence  TEXT NOT NULL,
  severity    TEXT NOT NULL,
  evidence    TEXT NOT NULL,
  fix         TEXT NOT NULL,
  suppressed_by TEXT,
  ts          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  request_id  TEXT NOT NULL REFERENCES requests(id),
  seq         INTEGER NOT NULL,
  offset_ms   INTEGER NOT NULL,
  size        INTEGER NOT NULL,
  PRIMARY KEY (request_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_diagnoses_request ON diagnoses(request_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_rule ON diagnoses(rule_id);
"""


class Store:
    """One connection guarded by a lock. Writes are short; async callers use `to_thread`."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes ----------------------------------------------------------

    def write_request(self, rec: RequestRecord, record_chunks: bool = False) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO requests
                   (id, ts, endpoint, model, request_body, request_headers,
                    response_body, status_code, stream, prompt_tokens,
                    completion_tokens, finish_reason, num_ctx, num_ctx_source,
                    ttft_ms, total_ms, chunk_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.id,
                    rec.ts,
                    rec.endpoint,
                    rec.model,
                    rec.request_body,
                    json.dumps(rec.request_headers, ensure_ascii=False),
                    rec.response_body,
                    rec.status_code,
                    1 if rec.stream else 0,
                    rec.usage.prompt_tokens,
                    rec.usage.completion_tokens,
                    rec.usage.finish_reason,
                    rec.num_ctx,
                    rec.num_ctx_source,
                    rec.ttft_ms,
                    rec.total_ms,
                    rec.chunk_count,
                ),
            )
            if record_chunks and rec.chunks:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO chunks (request_id, seq, offset_ms, size) VALUES (?,?,?,?)",
                    [(rec.id, c.seq, c.offset_ms, c.size) for c in rec.chunks],
                )
            self._conn.commit()

    def write_diagnoses(self, diagnoses: list[Diagnosis]) -> None:
        if not diagnoses:
            return
        with self._lock:
            self._conn.executemany(
                """INSERT INTO diagnoses
                   (request_id, rule_id, confidence, severity, evidence, fix, suppressed_by, ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (
                        d.request_id,
                        d.rule_id,
                        d.confidence,
                        d.severity,
                        json.dumps(d.evidence, ensure_ascii=False),
                        d.fix,
                        d.suppressed_by,
                        d.ts,
                    )
                    for d in diagnoses
                ],
            )
            self._conn.commit()

    async def awrite(self, rec: RequestRecord, diagnoses: list[Diagnosis], record_chunks: bool) -> None:
        def _write() -> None:
            self.write_request(rec, record_chunks)
            self.write_diagnoses(diagnoses)

        await asyncio.to_thread(_write)

    # --- reads ------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(sql, params).fetchall()

    def recent(
        self,
        limit: int = 20,
        model: str | None = None,
        rule: str | None = None,
        endpoint: str | None = None,
        only_findings: bool = True,
    ) -> list[sqlite3.Row]:
        """Most recent requests, newest first."""
        clauses: list[str] = []
        params: list = []
        if model:
            clauses.append("r.model LIKE ?")
            params.append(f"%{model}%")
        if endpoint:
            clauses.append("r.endpoint = ?")
            params.append(endpoint)
        if rule:
            clauses.append("EXISTS (SELECT 1 FROM diagnoses d WHERE d.request_id = r.id AND d.rule_id = ?)")
            params.append(rule.upper())
        elif only_findings:
            clauses.append("EXISTS (SELECT 1 FROM diagnoses d WHERE d.request_id = r.id)")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        return self.query(
            f"""SELECT r.*,
                       (SELECT COUNT(*) FROM diagnoses d WHERE d.request_id = r.id) AS diagnosis_count
                FROM requests r {where}
                ORDER BY r.ts DESC, r.id DESC
                LIMIT ?""",
            tuple(params),
        )

    def find_request(self, ident: str) -> sqlite3.Row | None:
        """Look a request up by full id, or by a unique prefix or suffix.

        Ids are ULID-like: the first half is a timestamp, so short prefixes
        collide. `localdoctor log` therefore prints the tail, and a tail lookup
        is what usually lands here.
        """
        ident = ident.strip().upper()
        if not ident:
            return None
        exact = self.query("SELECT * FROM requests WHERE id = ?", (ident,))
        if exact:
            return exact[0]
        for pattern in (f"%{ident}", f"{ident}%"):
            rows = self.query("SELECT * FROM requests WHERE id LIKE ?", (pattern,))
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                raise AmbiguousId(ident, [row["id"] for row in rows])
        return None

    def diagnoses_for(self, request_id: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM diagnoses WHERE request_id = ? ORDER BY id", (request_id,)
        )

    def chunks_for(self, request_id: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM chunks WHERE request_id = ? ORDER BY seq", (request_id,)
        )

    def stats(self) -> dict:
        """Counts for the dashboard header."""
        total = self.query("SELECT COUNT(*) AS n FROM requests")[0]["n"]
        flagged = self.query(
            "SELECT COUNT(DISTINCT request_id) AS n FROM diagnoses WHERE suppressed_by IS NULL"
        )[0]["n"]
        by_rule = self.query(
            """SELECT rule_id, confidence, COUNT(*) AS n FROM diagnoses
               GROUP BY rule_id, confidence ORDER BY rule_id, confidence"""
        )
        return {
            "total_requests": total,
            "flagged_requests": flagged,
            "by_rule": [dict(row) for row in by_rule],
        }


class AmbiguousId(LookupError):
    def __init__(self, ident: str, matches: list[str]) -> None:
        super().__init__(f"{ident} matches {len(matches)} requests")
        self.ident = ident
        self.matches = matches
