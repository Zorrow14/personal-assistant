# Jarvis

A voice-driven personal assistant that interprets natural-language commands, acts
on them through LLM tool-calling, speaks its replies, and journals every task into
an Obsidian vault as plain Markdown.

**Status: Phase 4 (memory).** Type a command, use push-to-talk, or just say
"Hey Jarvis" and speak. The LLM (Google Gemini) decides which tool to call,
Jarvis runs it and answers in text or aloud, and each task is journaled to the
vault. Jarvis can also search its own history: ask "what did I ask you to do
about X?" and it looks up the relevant past notes and cites them as
`[[note_name]]`. The wake word (openWakeWord), end-of-speech detection
(webrtcvad), speech-to-text (faster-whisper), text-to-speech (Piper, or the OS
voice) and the memory index (fastembed + Chroma) all run locally and cost
nothing. Barge-in, more tools and a GUI are not built yet; they are marked
`TODO(phase-N)` in the code.

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
| `JARVIS_SAMPLE_RATE` | no | `16000` | Microphone rate in Hz. Whisper needs 16 kHz; other rates are resampled. |
| `JARVIS_INPUT_DEVICE` / `JARVIS_OUTPUT_DEVICE` | no | system default | Device indices from `--list-devices`. |
| `JARVIS_STT_MODEL` | no | `base.en` | faster-whisper model size (`tiny.en`, `base.en`, `small.en`, ...). |
| `JARVIS_STT_DEVICE` | no | `cpu` | `cpu`, or `cuda` for an NVIDIA GPU. |
| `JARVIS_STT_COMPUTE_TYPE` | no | `int8` | `int8` for CPU, `float16` for GPU. |
| `JARVIS_TTS_PROVIDER` | no | `piper` | `piper` (neural voice, needs a voice file) or `pyttsx3` (OS voice, no setup). |
| `JARVIS_TTS_VOICE` | for piper | – | Path to a Piper `.onnx` voice. Its `.onnx.json` must sit beside it. |
| `JARVIS_WAKE_WORD_MODEL` | no | `hey_jarvis` | openWakeWord pretrained model name, or a path to a custom `.onnx`. |
| `JARVIS_WAKE_WORD_THRESHOLD` | no | `0.5` | Wake score, from 0 to 1. Higher means fewer false triggers but more misses. |
| `JARVIS_VAD_AGGRESSIVENESS` | no | `2` | webrtcvad strictness, 0–3. Raise it in noisy rooms. |
| `JARVIS_VAD_SILENCE_MS` | no | `800` | Trailing silence that ends a command. Raise it if you get cut off. |
| `JARVIS_VAD_FRAME_MS` | no | `30` | VAD frame length: 10, 20 or 30 ms. |
| `JARVIS_COMMAND_MAX_SECONDS` | no | `15` | Hard cap on one command's length. |
| `JARVIS_WAKE_CHIME` | no | `true` | Play a short cue when the wake word is heard. |
| `JARVIS_EMBEDDER_PROVIDER` | no | `local` | Embedding backend. Only `local` (on-device fastembed) exists, so note text never leaves the machine. |
| `JARVIS_EMBED_MODEL` | no | `BAAI/bge-small-en-v1.5` | fastembed model. It downloads once (about 70 MB) into `.jarvis/fastembed`. Changing it rebuilds the index. |
| `JARVIS_VECTOR_STORE` | no | `chroma` | Vector store. Only `chroma` (embedded, on disk) exists. |
| `JARVIS_CHROMA_PATH` | no | `.jarvis/chroma` | Where the index lives, relative to where you run Jarvis. The manifest and model cache sit next to it. |
| `JARVIS_RAG_TOP_K` | no | `5` | Chunks `search_memory` retrieves by default. |
| `JARVIS_RAG_CHUNK_CHARS` / `JARVIS_RAG_CHUNK_OVERLAP` | no | `1000` / `150` | Chunk size and overlap in characters. Changing either rebuilds the index. |
| `JARVIS_AUTO_INDEX` | no | `true` | Index notes written during a turn straight after it, so they're searchable immediately. |

`.env` is git-ignored. Never commit real keys.

### Voice setup

Download a Piper voice into `voices/` (git-ignored):

```sh
uv run python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

Then set `JARVIS_TTS_VOICE=voices/en_US-lessac-medium.onnx`. To skip this step,
set `JARVIS_TTS_PROVIDER=pyttsx3` and Jarvis will use the built-in OS voice.

The first `--voice` run downloads the Whisper model: about 145 MB for `base.en`,
cached under `~/.cache/huggingface`. After that, voice mode works offline, apart
from the Gemini calls.

## Run

```sh
uv run python -m jarvis.cli                 # text chat; type exit or quit, or press Ctrl-C, to leave
uv run python -m jarvis.cli --voice         # push-to-talk voice chat
uv run python -m jarvis.cli --wake          # hands-free: "Hey Jarvis", then your command
uv run python -m jarvis.cli --reindex       # build/refresh the memory index over existing notes
uv run python -m jarvis.cli --health        # check config + vault, then exit
uv run python -m jarvis.cli --list-devices  # audio device indices
uv run pytest                               # offline: fakes for LLM/STT/TTS/wake/VAD/memory, temp vault, no mic
```

In hands-free mode (`--wake`), no keys are needed:

1. Say **"Hey Jarvis"** and wait for the chime.
2. Say your command, then stop talking. Recording ends after 0.8 s of silence.
3. Jarvis shows the transcript, runs the command, and speaks the reply.
4. It goes back to listening for the wake word.

Say "Hey Jarvis… exit", or press Ctrl-C, to quit. The wake-word model ships
with openWakeWord, so there is nothing extra to download. While Jarvis is
speaking it doesn't listen for the wake word, and anything the mic picks up
during that time, including Jarvis's own voice, is thrown away.

In push-to-talk mode (`--voice`):

1. Press **Enter** and speak.
2. Press **Enter** again to stop recording.
3. Jarvis shows the transcript, runs the command, prints the reply and says it aloud.

You can also type a message at the prompt instead of speaking; the reply is
still spoken. To leave, say or type "exit" or "quit", or press Ctrl-C.

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

Voice is a thin layer around the same `Agent.run`. `core/voice_session.py`
transcribes the recording (`STTEngine`), calls the agent exactly as text mode
does, and speaks the reply (`TTSEngine`), with Markdown symbols removed first.
Blocking model and audio work runs through `asyncio.to_thread`. Each engine's
vendor code stays in its own module: `stt/whisper_stt.py`,
`tts/piper_tts.py`, `tts/system_tts.py` and `audio/io.py`.

Hands-free mode adds a small state machine, `core/voice_loop.py`:

```
IDLE ──wake word──▶ LISTENING ──silence or cap──▶ PROCESSING ──text──▶ SPEAKING ──▶ IDLE
                                                       └── empty transcript ──▶ IDLE
```

One 16 kHz `MicStream` feeds both consumers. The wake detector reads it in
80 ms frames and the VAD in 30 ms frames. After a trigger, the detector is
reset and fed a moment of silence. Without that, openWakeWord's rolling feature
window still holds "Hey Jarvis" and fires again immediately (a measured score of
0.98 without the flush, 0.00 with it). The loop reports LLM errors and failed
turns and keeps listening, but it stops after 3 failures in a row rather than
spinning on a broken device.

### Memory (retrieval)

Run `--reindex` once to index your existing notes. After that:

- **`search_memory`** is a read-only tool. It embeds the question and finds the
  nearest note chunks. It returns one entry per note: the `[[note_name]]`, the
  date, the tags, a relevance score and a snippet. It also states today's date
  so the model can work out relative dates like "last week". The model is told
  to answer only from these results and to say so when nothing fits.
- **Indexing** reads everything under `<vault>/Jarvis/`. The text it embeds for
  each note is the frontmatter `command` plus the note body. That text is split
  into chunks with ids like `Tasks/<file>.md::0`, so re-indexing replaces chunks
  rather than duplicating them.
- **The manifest** (`.jarvis/index_manifest.json`) keeps a content hash for each
  note. Unchanged notes are skipped, edited notes are re-embedded (stale chunks
  are removed), and deleted notes are dropped. If the embedding model, the
  chunking settings or the vault change, the whole index is rebuilt.
- **Auto-index:** `AutoIndexingAgent` is a subclass of `Agent`. It records each
  note's size and modified time before a turn, then indexes whatever changed
  once the turn is done. A task you log now can be found in the very next turn,
  in every mode, without touching the vault writer or the tools.

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
├── cli.py                entrypoint: text REPL, --voice, --health, wiring
├── core/
│   ├── interfaces.py     LLMClient / STTEngine / TTSEngine ABCs + neutral types
│   ├── agent.py          tool-calling loop + confirmation gate
│   ├── voice_session.py  one push-to-talk turn: transcribe -> agent -> speak
│   └── voice_loop.py     hands-free state machine (IDLE/LISTENING/PROCESSING/SPEAKING)
├── audio/
│   ├── io.py             mic (Enter-to-stop recording, shared MicStream), playback, chime
│   └── vad.py            webrtcvad end-of-command detection
├── wakeword/openwakeword_detector.py  OpenWakeWordDetector
├── stt/whisper_stt.py    WhisperSTT (faster-whisper)
├── tts/
│   ├── factory.py        tts_provider -> TTSEngine
│   ├── piper_tts.py      PiperTTS
│   └── system_tts.py     Pyttsx3TTS (OS voice)
├── llm/
│   ├── factory.py        llm_provider -> LLMClient
│   └── gemini_client.py  GeminiClient (the only Gemini-aware module)
├── memory/
│   ├── vault.py          Obsidian vault writer + path guard
│   ├── documents.py      frontmatter parsing + chunking
│   ├── indexer.py        VaultIndexer (incremental, manifest-based)
│   ├── embedder.py       LocalEmbedder (fastembed)
│   ├── vector_store.py   ChromaStore (chromadb)
│   ├── auto_index.py     AutoIndexingAgent (index after each turn)
│   └── factory.py        settings -> embedder + store + indexer
└── tools/
    ├── base.py           Tool ABC (pydantic args_model) + ToolRegistry
    ├── vault_tools.py    WriteTaskNoteTool
    └── memory_tools.py   SearchMemoryTool
```

To add an LLM provider, implement `LLMClient` in `llm/` and add a branch to
`llm/factory.py`. To add a voice engine, implement `STTEngine` or `TTSEngine`
and add a branch to `tts/factory.py`. Nothing else needs to change.

## Roadmap

- **Phase 5:** more tools and plugin auto-discovery.
- **Phase 6:** a GUI.
- **Later:** barge-in (interrupting Jarvis mid-reply with the wake word); an opt-in cloud embedder; pgvector behind the `VectorStore` interface; date filters in `search_memory`; trimming long conversation histories.
- **Phase 5:** more tools and plugin auto-discovery.
