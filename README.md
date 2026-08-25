# LocalDoctor

```
When a prompt does not fit the context window, Ollama truncates it
and then returns HTTP 200. Nothing in the response says so.

LocalDoctor catches this and similar silent failures — with the evidence.
```

It sits between your application and your local LLM server, watches the traffic,
and tells you when a request quietly went wrong. Existing tools are *observers*
(graphs, tokens/sec, logs). LocalDoctor is a *diagnostician*: "this signal means
this, and here is the fix."

## Install

Not on PyPI yet. Install straight from the repository:

```bash
pip install git+https://github.com/klncgty/localdoctor.git
```

Or run it without installing anything, with [uv](https://docs.astral.sh/uv/):

```bash
uvx --from git+https://github.com/klncgty/localdoctor.git localdoctor serve
```

## Use

Start the proxy and point your application's base URL at it. That is the whole
integration — no code changes, no SDK, no configuration file.

```bash
localdoctor serve
```

```
localdoctor 0.1.0  http://127.0.0.1:11435  →  http://localhost:11434
  recording to ~/.localdoctor/localdoctor.db   ·   silence means nothing is wrong
```

```diff
- base_url = "http://localhost:11434"
+ base_url = "http://localhost:11435"
```

Healthy requests print nothing. When something is wrong you get evidence and a
prescription:

```
⚠  CONTEXT LIMIT EXCEEDED   qwen3:14b   14:32:07   confidence: high
   Model read            4,095 tokens
   Window limit          4,096  (source: model default)
   Sent (lower bound)    >= 5,000 tokens
   Input appears to exceed the context limit; what was cut is not visible from here.
   ► Try num_ctx=16384 (this model supports 262,144), or split the input.
   related: R003
```

You can also check your setup without waiting for traffic:

```bash
localdoctor doctor
```

### Options

```
localdoctor serve [--port 11435] [--upstream http://localhost:11434]
                  [--host 127.0.0.1] [--quiet] [--record] [--db PATH]
```

`--quiet` prints nothing but still records everything. `--record` additionally
stores chunk-level timing.

## What it detects

| Rule | Condition | Confidence |
|---|---|---|
| **R001** Context limit exceeded | prompt_eval_count pinned to the window, or the input provably could not fit *and* the model read less than was sent | never `certain` — the signal has two explanations |
| **R002** Context underuse | the window in use is under a quarter of what the model supports | informational, printed once per model |
| **R003** Empty output | HTTP 200, tokens generated, content empty | `certain` |
| **R004** Reasoning budget starvation | the reasoning block consumed the generation budget and no answer was left | `high` |

When several rules fire on one request, the **root cause** is printed and the
rest are recorded as related. R001 suppresses R003 and R004; R004 suppresses
R003.

Truncation does not always look the same. Ollama runs llama-server with
`--context-shift`, which drops part of an oversized prompt rather than filling
the window, so the reported token count can land far below `num_ctx`. R001
catches both shapes.

## Confidence

Every claim carries the strength of its evidence:

| Level | Meaning | Behaviour |
|---|---|---|
| `certain` | direct evidence, no other explanation | printed |
| `high` | two independent signals agree | printed |
| `medium` | one strong signal, alternatives possible | printed as "likely" |
| `low` | heuristic only | **never printed**, recorded only |

Token counts are never guessed as a point value. LocalDoctor computes a
conservative **lower bound** (`len(text) / 12`), so when it says the input
exceeded the window, that is proven arithmetic rather than an estimate. It makes
no claim in the other direction.

When the effective context window cannot be determined — Ollama's server-wide
`OLLAMA_CONTEXT_LENGTH` is not visible to a proxy — LocalDoctor says so and caps
any diagnosis built on the estimate at `medium`.

## Data

Every request is recorded, including the ones that produced no diagnosis, so
"why didn't it warn me?" always has an answer. Raw request bodies and headers
are stored so recorded traffic can be replayed later.

Default location: `~/.localdoctor/localdoctor.db` (SQLite).

## Principles

1. **Zero code changes.** You only change a base URL.
2. **Never modify the request.** Requests and responses pass through byte for byte.
3. **Never claim what you do not know.** Every claim has a signal behind it.
4. **A verdict, not a graph.** Evidence and a prescription, not a dashboard.
5. **Zero configuration.** Install, run one command, done.
6. **Fully local.** No outbound network calls. No telemetry.

Works with Ollama's native API (`/api/chat`, `/api/generate`, `/api/embed`) and
with any OpenAI-compatible server — llama.cpp, vLLM, LM Studio — through
`/v1/chat/completions` and `/v1/completions`. Everything else is dumb
passthrough.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite needs neither a GPU nor a running Ollama: `tests/fake_ollama.py`
reproduces the response formats, including the failure modes.
