# Jarvis

A voice-driven personal assistant that interprets natural-language commands, acts
on them through LLM tool-calling, speaks its replies, and journals every task into
an Obsidian vault as plain Markdown.

**Status: Phase 1 (text MVP).** You type a command in the terminal. The LLM
(Google Gemini) decides which tool to call, Jarvis runs it and replies, and each
task is journaled to the vault. Voice, semantic search and extra tools are not
built yet; they are marked `TODO(phase-N)` in the code.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for the project if needed.
- An Obsidian vault folder, or any existing folder you want to use as one.
- A free Gemini API key from <https://aistudio.google.com/apikey>.

## Setup

```sh
uv sync                  # create .venv and install dependencies
cp .env.example .env     # PowerShell: Copy-Item .env.example .env
```

Then edit `.env` and set `JARVIS_VAULT_PATH` and `LLM_API_KEY`.

### Environment variables

Variables are read from the process environment first, then from `.env` in the
directory you run Jarvis from.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `JARVIS_VAULT_PATH` | yes | – | Root of your Obsidian vault. Must already exist. Jarvis writes only under `<vault>/Jarvis/`. |
| `LLM_API_KEY` | for chat | – | Gemini API key. Held as a secret, so it is never logged. `JARVIS_LLM_API_KEY` also works. |
| `JARVIS_LOG_LEVEL` | no | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. Use `WARNING` for a quiet chat. |
| `JARVIS_LLM_PROVIDER` | no | `gemini` | Which `LLMClient` to use. Only `gemini` exists so far. |
| `JARVIS_LLM_MODEL` | no | `gemini-2.5-flash` | Model name, as listed in AI Studio. |
| `JARVIS_LLM_MAX_TOKENS` | no | `1024` | Output token cap per reply. |
| `JARVIS_LLM_TEMPERATURE` | no | `0.7` | Sampling temperature, from 0 to 2. |
| `JARVIS_AGENT_MAX_ITERATIONS` | no | `8` | Maximum LLM round-trips per command. |

`.env` is git-ignored. Never commit real keys.

## Run

```sh
uv run python -m jarvis.cli            # chat; type exit or quit, or press Ctrl-C, to leave
uv run python -m jarvis.cli --health   # check config + vault, then exit
uv run pytest                          # offline: uses a fake LLM and a temporary vault
```

Example:

```
you> log that I finished the vault module
jarvis> Logged it — "Finish the vault module" is in your vault.
```

`--health` loads the settings, creates `<vault>/Jarvis/` if it is missing, checks
that the folder is writable, and prints `OK` with the resolved vault path and the
LLM settings. It never calls the LLM. It exits non-zero if anything fails.

## How it works

1. `cli.py` builds everything in one place: Settings, then logging, then an
   `LLMClient` chosen by `llm_provider`, then a `ToolRegistry`, then the `Agent`.
2. `Agent.run` sends the conversation and the tool schemas to the LLM. If the
   model asks for tools, the agent validates each call's arguments against the
   tool's pydantic model, runs the tool, adds the result to the conversation and
   asks the model again. It stops when the model replies in text, or after
   `agent_max_iterations` round-trips.
3. Tools with `requires_confirmation = True` ask `Allow? [y/N]` in the terminal
   before they run. Unknown tools, bad arguments and tool exceptions are sent
   back to the model as errors; they don't crash the chat.
4. The conversation history lives on the Agent, so follow-up questions have context.

The agent works only with the neutral types in `core/interfaces.py`. Everything
specific to Gemini lives in `llm/gemini_client.py`: role mapping, function
declarations, finish reasons, and retries. The client retries 429 and 5xx errors
up to 5 times, waiting 1, 2, 4 and then 8 seconds.

## Vault layout

```
<vault>/Jarvis/
├── Tasks/  YYYY-MM-DD-HHMM-<slug>.md   one note per task
└── Daily/  YYYY-MM-DD.md               timestamped log lines
```

A task note looks like this:

```markdown
---
date: 2026-09-25
command: "Turn on the living room lights"
status: completed
tags: [home, lights]
tools_used: [smart_home]
---

# Turn on the living room lights

Switched on 3 lights.

Related: [[Living Room]]
```

Every path goes through a guard, `Vault._safe_path`, that resolves `..` and
symlinks. Any write that would land outside `<vault>/Jarvis/` raises
`VaultPathError`. If two notes share a name, the new one gets a numeric suffix
so nothing is overwritten.

## Project layout

```
src/jarvis/
├── config.py             Settings (pydantic-settings)
├── logging.py            structlog setup + get_logger()
├── cli.py                entrypoint: REPL, --health, wiring
├── core/
│   ├── interfaces.py     LLMClient / STTEngine / TTSEngine ABCs + neutral message types
│   └── agent.py          tool-calling loop + confirmation gate
├── llm/
│   ├── factory.py        llm_provider -> LLMClient
│   └── gemini_client.py  GeminiClient (the only Gemini-aware module)
├── memory/vault.py       Obsidian vault writer + path guard
└── tools/
    ├── base.py           Tool ABC (pydantic args_model) + ToolRegistry
    └── vault_tools.py    WriteTaskNoteTool
```

To add a provider, implement `LLMClient` in `llm/` and add a branch to
`llm/factory.py`. Nothing else needs to change.

## Roadmap

- **Phases 2–3:** voice: speech-to-text, text-to-speech and a wake word.
- **Phase 4:** semantic search over the vault (`search_vault` is filename-only until then), plus trimming of long conversation histories.
- **Phase 5:** more tools and plugin auto-discovery.
