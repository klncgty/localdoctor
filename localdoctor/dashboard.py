"""Embedded web dashboard (phase 2).

Self-contained on purpose: the page carries its own CSS and JS, loads no font,
no script and no stylesheet from anywhere else. Nothing leaves this machine.

It shows verdicts, not graphs — the same evidence the terminal prints, plus the
`low` confidence records the terminal deliberately withholds.
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from localdoctor.store import AmbiguousId, Store

BODY_PREVIEW_BYTES = 20_000


def _request_dict(row: sqlite3.Row, with_bodies: bool = False) -> dict:
    data = {
        "id": row["id"],
        "short_id": row["id"][-8:],
        "ts": row["ts"],
        "endpoint": row["endpoint"],
        "model": row["model"],
        "status_code": row["status_code"],
        "stream": bool(row["stream"]),
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "finish_reason": row["finish_reason"],
        "num_ctx": row["num_ctx"],
        "num_ctx_source": row["num_ctx_source"],
        "ttft_ms": row["ttft_ms"],
        "total_ms": row["total_ms"],
        "chunk_count": row["chunk_count"],
    }
    if with_bodies:
        data["request_body"] = _decode(row["request_body"])
        data["response_body"] = _decode(row["response_body"])
    return data


def _decode(raw: bytes | None) -> str | None:
    if not raw:
        return None
    text = raw[:BODY_PREVIEW_BYTES].decode("utf-8", errors="replace")
    if len(raw) > BODY_PREVIEW_BYTES:
        text += f"\n… {len(raw) - BODY_PREVIEW_BYTES} more bytes"
    return text


def _diagnosis_dict(row: sqlite3.Row) -> dict:
    try:
        evidence = json.loads(row["evidence"])
    except Exception:
        evidence = {}
    return {
        "rule_id": row["rule_id"],
        "confidence": row["confidence"],
        "severity": row["severity"],
        "evidence": evidence,
        "fix": row["fix"],
        "suppressed_by": row["suppressed_by"],
        "ts": row["ts"],
    }


def create_dashboard_app(store: Store) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE

    @app.get("/api/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(store.stats())

    @app.get("/api/requests")
    async def requests(
        limit: int = 100,
        model: str | None = None,
        rule: str | None = None,
        all: bool = False,
    ) -> JSONResponse:
        rows = store.recent(
            limit=min(limit, 500), model=model, rule=rule, only_findings=not all
        )
        out = []
        for row in rows:
            data = _request_dict(row)
            data["diagnoses"] = [_diagnosis_dict(d) for d in store.diagnoses_for(row["id"])]
            out.append(data)
        return JSONResponse(out)

    @app.get("/api/requests/{ident}")
    async def request_detail(ident: str) -> JSONResponse:
        try:
            row = store.find_request(ident)
        except AmbiguousId as exc:
            return JSONResponse({"error": str(exc), "matches": exc.matches}, status_code=409)
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = _request_dict(row, with_bodies=True)
        data["diagnoses"] = [_diagnosis_dict(d) for d in store.diagnoses_for(row["id"])]
        data["chunks"] = [dict(c) for c in store.chunks_for(row["id"])]
        return JSONResponse(data)

    return app


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalDoctor</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161922; --line: #262b36; --text: #d6dae3;
    --dim: #7b8494; --cyan: #6cc4d8; --red: #e56a6a; --amber: #d8ae5f;
    --green: #74c48b;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f7f8fa; --panel: #ffffff; --line: #e2e5eb; --text: #1c2028;
      --dim: #6b7280; --cyan: #0f7d95; --red: #b83232; --amber: #8a6212;
      --green: #2f7d47;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  header {
    padding: 18px 24px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap;
  }
  h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .02em; }
  .stat { color: var(--dim); font-size: 13px; }
  .stat b { color: var(--text); font-weight: 600; }
  main { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,420px); gap: 0; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .list { border-right: 1px solid var(--line); min-height: 70vh; }
  .controls { padding: 12px 24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  input, select, button {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 5px; padding: 5px 9px; font: inherit; font-size: 13px;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--dim); }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; font-weight: 500; color: var(--dim); font-size: 12px;
    padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap;
  }
  td { padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  td:first-child, th:first-child { padding-left: 24px; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: var(--panel); }
  tbody tr.on { background: var(--panel); }
  .id { color: var(--cyan); }
  .dim { color: var(--dim); }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .tag { font-size: 12px; padding: 1px 6px; border-radius: 4px; border: 1px solid; }
  .c-certain, .c-high { color: var(--red); border-color: var(--red); }
  .c-medium { color: var(--amber); border-color: var(--amber); }
  .c-low, .sup { color: var(--dim); border-color: var(--line); }
  .detail { padding: 18px 24px; overflow-x: auto; }
  .detail h2 { font-size: 13px; margin: 0 0 12px; color: var(--dim); font-weight: 500; }
  .kv { display: grid; grid-template-columns: 110px 1fr; gap: 3px 12px; margin-bottom: 20px; }
  .kv div:nth-child(odd) { color: var(--dim); }
  .diag { border: 1px solid var(--line); border-radius: 6px; padding: 12px; margin-bottom: 12px; }
  .diag.suppressed { opacity: .55; }
  .diag-head { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  .ev { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 13px; }
  .ev div:nth-child(odd) { color: var(--dim); }
  .fix { margin-top: 10px; color: var(--green); font-size: 13px; }
  pre {
    background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
    padding: 10px; overflow-x: auto; font-size: 12px; margin: 0 0 16px;
    white-space: pre-wrap; word-break: break-word; max-height: 260px;
  }
  .empty { padding: 40px 24px; color: var(--dim); }
</style>
</head>
<body>
<header>
  <h1>LocalDoctor</h1>
  <span class="stat"><b id="s-total">0</b> requests recorded</span>
  <span class="stat"><b id="s-flagged">0</b> with findings</span>
  <span class="stat" id="s-rules"></span>
</header>
<main>
  <section class="list">
    <div class="controls">
      <input id="f-model" placeholder="filter by model" size="18">
      <select id="f-rule">
        <option value="">any rule</option>
        <option>R001</option><option>R002</option><option>R003</option><option>R004</option>
      </select>
      <label class="dim"><input type="checkbox" id="f-all"> include healthy</label>
      <button id="refresh">refresh</button>
    </div>
    <table>
      <thead><tr>
        <th>id</th><th>time</th><th>model</th><th>endpoint</th>
        <th class="num">tokens</th><th class="num">ms</th><th>findings</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="empty" id="empty" hidden>Nothing recorded yet. Silence means nothing is wrong.</div>
  </section>
  <aside class="detail" id="detail">
    <div class="dim">Select a request.</div>
  </aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "—").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const num = (n) => (n === null || n === undefined) ? "—" : Number(n).toLocaleString("en-US");
const clock = (ts) => { try { return new Date(ts).toLocaleTimeString(); } catch { return ts; } };

function tags(diagnoses) {
  if (!diagnoses.length) return '<span class="dim">—</span>';
  return diagnoses.map(d => d.suppressed_by
    ? `<span class="tag sup">${esc(d.rule_id)} &darr;${esc(d.suppressed_by)}</span>`
    : `<span class="tag c-${esc(d.confidence)}">${esc(d.rule_id)} ${esc(d.confidence)}</span>`
  ).join(" ");
}

async function loadStats() {
  const s = await (await fetch("/api/stats")).json();
  $("s-total").textContent = num(s.total_requests);
  $("s-flagged").textContent = num(s.flagged_requests);
  $("s-rules").textContent = s.by_rule.map(r => `${r.rule_id} ${r.confidence} ×${r.n}`).join("   ");
}

async function loadList() {
  const params = new URLSearchParams();
  if ($("f-model").value) params.set("model", $("f-model").value);
  if ($("f-rule").value) params.set("rule", $("f-rule").value);
  if ($("f-all").checked) params.set("all", "true");
  const rows = await (await fetch("/api/requests?" + params)).json();
  $("empty").hidden = rows.length > 0;
  $("rows").innerHTML = rows.map(r => `
    <tr data-id="${esc(r.id)}">
      <td class="id">${esc(r.short_id)}</td>
      <td class="dim">${esc(clock(r.ts))}</td>
      <td>${esc(r.model)}</td>
      <td class="dim">${esc(r.endpoint)}</td>
      <td class="num">${num(r.prompt_tokens)} &rarr; ${num(r.completion_tokens)}</td>
      <td class="num dim">${num(r.total_ms)}</td>
      <td>${tags(r.diagnoses)}</td>
    </tr>`).join("");
  for (const tr of $("rows").querySelectorAll("tr")) {
    tr.onclick = () => { select(tr); loadDetail(tr.dataset.id); };
  }
}

function select(tr) {
  for (const other of $("rows").querySelectorAll("tr.on")) other.classList.remove("on");
  tr.classList.add("on");
}

function evidenceRows(evidence) {
  return Object.entries(evidence).map(([k, v]) =>
    `<div>${esc(k)}</div><div>${esc(Array.isArray(v) ? v.join(", ") : v)}</div>`).join("");
}

async function loadDetail(id) {
  const r = await (await fetch("/api/requests/" + encodeURIComponent(id))).json();
  if (r.error) { $("detail").innerHTML = `<div class="dim">${esc(r.error)}</div>`; return; }
  const diagnoses = r.diagnoses.length ? r.diagnoses.map(d => `
    <div class="diag ${d.suppressed_by ? "suppressed" : ""}">
      <div class="diag-head">
        <b>${esc(d.rule_id)}</b>
        <span class="tag c-${esc(d.confidence)}">${esc(d.confidence)}</span>
        <span class="dim">${esc(d.severity)}</span>
        ${d.suppressed_by ? `<span class="dim">suppressed by ${esc(d.suppressed_by)}</span>` : ""}
        ${d.confidence === "low" ? `<span class="dim">recorded only, never printed live</span>` : ""}
      </div>
      <div class="ev">${evidenceRows(d.evidence)}</div>
      <div class="fix">&#9658; ${esc(d.fix)}</div>
    </div>`).join("") : `<div class="dim">No diagnoses — this request looked healthy.</div>`;

  $("detail").innerHTML = `
    <h2>${esc(r.id)}</h2>
    <div class="kv">
      <div>time</div><div>${esc(new Date(r.ts).toLocaleString())}</div>
      <div>endpoint</div><div>${esc(r.endpoint)} &nbsp;<span class="dim">stream: ${r.stream ? "yes" : "no"}</span></div>
      <div>model</div><div>${esc(r.model)}</div>
      <div>status</div><div>${esc(r.status_code)} &nbsp;<span class="dim">ttft ${num(r.ttft_ms)}ms · total ${num(r.total_ms)}ms · ${num(r.chunk_count)} chunks</span></div>
      <div>tokens</div><div>${num(r.prompt_tokens)} &rarr; ${num(r.completion_tokens)} &nbsp;<span class="dim">finish: ${esc(r.finish_reason)}</span></div>
      <div>window</div><div>${num(r.num_ctx)} &nbsp;<span class="dim">(${esc(r.num_ctx_source)})</span></div>
    </div>
    ${diagnoses}
    <h2>request body</h2><pre>${esc(r.request_body)}</pre>
    <h2>response body</h2><pre>${esc(r.response_body)}</pre>`;
}

async function refresh() { await loadStats(); await loadList(); }
$("refresh").onclick = refresh;
$("f-model").oninput = loadList;
$("f-rule").onchange = loadList;
$("f-all").onchange = loadList;
refresh();
</script>
</body>
</html>
"""
