"""Test doubles that implement Jarvis interfaces without any network access."""

import re
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from jarvis.core.interfaces import (
    AudioSamples,
    Embedder,
    FrameSource,
    LLMClient,
    LLMResponse,
    Message,
    MetadataValue,
    SearchHit,
    STTEngine,
    ToolSpec,
    TTSEngine,
    VectorStore,
    WakeWordDetector,
)
from jarvis.tools.base import Tool


class FakeSTTEngine(STTEngine):
    """Returns scripted transcripts in order and records the audio it received."""

    def __init__(self, transcripts: Sequence[str]) -> None:
        self._transcripts = list(transcripts)
        self.received: list[AudioSamples] = []

    def transcribe(self, audio: AudioSamples) -> str:
        self.received.append(audio)
        return self._transcripts.pop(0)


class FakeTTSEngine(TTSEngine):
    """Records everything it was asked to speak."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools) if tools is not None else None))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class EchoArgs(BaseModel):
    text: str


class RecordingTool(Tool):
    """Echoes its input and records each run."""

    name = "echo"
    description = "Echo the text back."
    args_model = EchoArgs

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> str:
        self.runs.append(kwargs)
        return f"echo: {kwargs['text']}"


class DangerousTool(RecordingTool):
    """Same as RecordingTool but gated behind user confirmation."""

    name = "delete_everything"
    description = "Pretend to delete everything."
    requires_confirmation = True


class ExplodingTool(RecordingTool):
    name = "explode"
    description = "Always fails."

    async def run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")


class FakeWakeWordDetector(WakeWordDetector):
    """Returns scripted wake scores, one per frame, and records resets."""

    def __init__(self, scores: Sequence[float], frame_samples: int = 1280) -> None:
        self._scores = list(scores)
        self._frame_samples = frame_samples
        self.frames_seen = 0
        self.resets = 0

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def process(self, frame: AudioSamples) -> float:
        assert len(frame) == self._frame_samples
        self.frames_seen += 1
        if not self._scores:
            raise AssertionError("FakeWakeWordDetector ran out of scripted scores")
        return self._scores.pop(0)

    def reset(self) -> None:
        self.resets += 1


class FakeFrameSource:
    """Endless silence, recording how much was read and how often it was cleared."""

    def __init__(self) -> None:
        self.samples_read = 0
        self.clears = 0

    def read(self, n_samples: int) -> AudioSamples:
        self.samples_read += n_samples
        return np.zeros(n_samples, dtype=np.float32)

    def clear(self) -> None:
        self.clears += 1


class FakeVAD:
    """A CommandRecorder that returns a fixed buffer without reading the source."""

    def __init__(self, seconds: float = 1.0, sample_rate: int = 16000) -> None:
        self.buffer = np.full(int(seconds * sample_rate), 0.1, dtype=np.float32)
        self.calls = 0

    def record_command(self, source: FrameSource) -> AudioSamples:
        self.calls += 1
        return self.buffer


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing vectors: similar wording -> similar vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.calls = 0
        self.texts_embedded = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self._dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self._dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()


class FakeVectorStore(VectorStore):
    """In-memory store with cosine similarity; records upserts and deletes."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[list[float], str, dict[str, MetadataValue]]] = {}
        self.upserted: list[str] = []
        self.deleted: list[str] = []
        self.resets = 0

    def upsert(self, ids, embeddings, documents, metadatas) -> None:  # type: ignore[no-untyped-def]
        for i, record_id in enumerate(ids):
            self.records[record_id] = (embeddings[i], documents[i], dict(metadatas[i]))
            self.upserted.append(record_id)

    def query(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        q = np.asarray(embedding)
        scored = []
        for record_id, (vec, doc, meta) in self.records.items():
            v = np.asarray(vec)
            denom = np.linalg.norm(q) * np.linalg.norm(v)
            scored.append(SearchHit(record_id, doc, meta, float(q @ v / denom) if denom else 0.0))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.deleted.append(record_id)
            self.records.pop(record_id, None)

    def count(self) -> int:
        return len(self.records)

    def reset(self) -> None:
        self.resets += 1
        self.records.clear()
