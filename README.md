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

## Look at what was recorded

The live terminal only shows what crosses the confidence threshold. Everything
else — including the `low` confidence guesses it deliberately withholds — is in
the database.

```bash
localdoctor log                 # requests that produced a finding
localdoctor log --all           # healthy ones too
localdoctor log --rule R001     # only this rule
localdoctor show 9HG3D0Y6       # everything about one request
```

```
id        time      model       endpoint    tokens  ms  findings
8Y3CW1ZQ  14:53:00  gemma3:4b   /api/chat    8 → 9   0  R001 low
9HG3D0Y6  14:53:00  qwen3.5:9b  /api/chat   8 → 24   0  R003↳R004  R004 high
JQTDS9MC  14:53:00  qwen3.5:9b  /api/chat  256 → 9   1  R001 high  R002 certain
```

`R003↳R004` means R003 fired but was suppressed by R004 as the root cause.
`R001 low` was recorded and never printed live.

There is a dashboard too. It carries its own CSS and JS and loads nothing from
anywhere else:

```bash
localdoctor dashboard           # http://127.0.0.1:11436
```

## Replay

Because every request is stored with its raw body, "does my agent still work if
I switch models?" becomes a command:

```bash
localdoctor replay JQTDS9MC -m gemma3:4b -m llama3.2:3b
```

```
replay  JQTDS9MC   /api/chat   recorded on qwen3.5:9b

model                  in → out  finish  ms  output  vs recorded  findings
qwen3.5:9b (recorded)   256 → 9  length   1   40 ch            —  R001 high  R002 certain
gemma3:4b               256 → 9  length   0   39 ch     86% same  R001 high  R002 certain
llama3.2:3b             256 → 9  length   0   41 ch     84% same  R001 high  R002 certain

  --- recorded qwen3.5:9b
  +++ gemma3:4b
  @@ -1 +1 @@
  -This is a normal answer from qwen3.5:9b.
  +This is a normal answer from gemma3:4b.
```

Every model is diagnosed independently, so a finding that follows the request
rather than the model — as R001 does above — is visible as such.

Replay changes only the `model` field of the recorded request. It never
modifies the stored record and never writes its results to the database.

### Options

```
localdoctor serve     [--port 11435] [--upstream http://localhost:11434]
                      [--host 127.0.0.1] [--quiet] [--record] [--db PATH]
localdoctor log       [--limit 20] [--model X] [--rule R001] [--all] [--db PATH]
localdoctor show      <id> [--full] [--db PATH]
localdoctor dashboard [--port 11436] [--host 127.0.0.1] [--db PATH]
localdoctor replay    <id> [--model X ...] [--upstream URL] [--no-diff] [--db PATH]
localdoctor doctor    [--upstream http://localhost:11434]
```

`--quiet` prints nothing but still records everything. `--record` additionally
stores chunk-level timing. Ids may be given in full or as any unique part —
the tail printed by `log` is enough.

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
are stored, which is what makes `localdoctor replay` possible.

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
