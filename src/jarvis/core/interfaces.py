"""Abstract engine interfaces and the vendor-neutral types they exchange.

The rest of Jarvis depends only on these types, never on a vendor SDK, so cloud
providers and local models can be swapped behind them. Provider clients map
these to and from their own wire formats.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

Role = Literal["user", "assistant", "tool"]

AudioSamples: TypeAlias = npt.NDArray[np.float32]
"""Mono float32 PCM samples in [-1, 1]."""

StopReason = Literal["end_turn", "tool_use", "max_tokens", "safety", "error", "other"]
"""Why the model stopped: finished its turn, wants tools run, hit the output
limit, was blocked by a safety filter, produced a malformed call, or anything else."""


class LLMError(Exception):
    """An LLM backend failed (after any retries) or is misconfigured."""

    def __init__(self, message: str = "", *, status_code: int | None = None) -> None:
        """
        Args:
            message: What went wrong.
            status_code: The provider's HTTP status, if any (429 = rate-limited,
                5xx = service trouble), so callers can tell transient failures apart.
        """
        super().__init__(message)
        self.status_code = status_code


class VoiceError(Exception):
    """A speech-to-text, text-to-speech or audio engine failed or is misconfigured."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the LLM to invoke a tool."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation with an LLM.

    - `user`: `content` is what the user said.
    - `assistant`: `content` is the model's text (may be empty) and
      `tool_calls` lists any tools it asked for.
    - `tool`: `content` is a tool's result; `tool_call_id` / `tool_name` link it
      to the ToolCall it answers.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    """For `tool` messages: the result describes a failure."""
    raw: object = None
    """For `assistant` messages: the provider's original response, opaque to
    everything except the client that produced it (lets it replay the turn
    losslessly). Other code must not inspect it."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Vendor-neutral description of a tool the LLM may call."""

    name: str
    description: str
    input_schema: dict[str, Any]
    """JSON Schema for the tool's arguments."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens one LLM request consumed, as reported by the provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    """Reasoning tokens some models spend before answering (billed like output)."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The result of a single LLM completion."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    raw: object = None
    """The provider's untouched response, for debugging and history replay."""
    usage: TokenUsage | None = None
    """Token counts for this request; None if the provider didn't report them."""


class LLMClient(ABC):
    """A chat-completion backend with tool-calling support."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send `messages` to the model and return its reply.

        Args:
            messages: Conversation so far, oldest first.
            tools: Tools the model may request; `None` disables tool use.

        Returns:
            The model's text and/or requested tool calls.

        Raises:
            LLMError: If the backend fails after any retries.
        """

    async def aclose(self) -> None:
        """Release network resources. The default does nothing."""


class STTEngine(ABC):
    """Speech-to-text engine."""

    @abstractmethod
    def transcribe(self, audio: AudioSamples) -> str:
        """Transcribe recorded speech into text. Blocking.

        Args:
            audio: Mono float32 samples in [-1, 1] at the sample rate the
                engine was configured with.

        Returns:
            The recognised text, stripped; empty if nothing was heard.
        """

    async def transcribe_async(self, audio: AudioSamples) -> str:
        """`transcribe` on a worker thread, so the event loop never stalls."""
        return await asyncio.to_thread(self.transcribe, audio)


class TTSEngine(ABC):
    """Text-to-speech engine."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak `text` aloud, blocking until playback finishes."""

    async def speak_async(self, text: str) -> None:
        """`speak` on a worker thread, so the event loop never stalls."""
        await asyncio.to_thread(self.speak, text)


class WakeWordDetector(ABC):
    """Streaming wake-word detector fed fixed-size frames of 16 kHz mono audio."""

    @property
    @abstractmethod
    def frame_samples(self) -> int:
        """Samples per frame that `process` expects."""

    @abstractmethod
    def process(self, frame: AudioSamples) -> float:
        """Feed one frame (`frame_samples` long) and return the current wake score, 0–1."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state after a trigger so the same audio can't fire again."""


class FrameSource(Protocol):
    """A live audio source that consumers read in frames of their chosen size."""

    def read(self, n_samples: int) -> AudioSamples:
        """Block until `n_samples` mono float32 samples are available and return them."""
        ...

    def clear(self) -> None:
        """Discard any audio buffered but not yet read."""
        ...


class CommandRecorder(Protocol):
    """Records one spoken command from a live source, deciding when it has ended."""

    def record_command(self, source: FrameSource) -> AudioSamples:
        """Record from now until the speaker stops (or a length cap), and return the audio."""
        ...


# --- Memory / retrieval (Phase 4) -------------------------------------------

MetadataValue: TypeAlias = str | int | float | bool
"""Metadata values every vector store can hold (no nested structures)."""


class RetrievalError(Exception):
    """An embedder, vector store or index failed or is misconfigured."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result from a vector search."""

    id: str
    document: str
    metadata: dict[str, MetadataValue]
    score: float
    """Similarity, higher is closer (cosine similarity for the default store)."""


class Embedder(ABC):
    """Turns text into fixed-length vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of every vector this embedder produces."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in one batch. Blocking."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Models trained with a query prefix override this."""
        return self.embed([text])[0]


class VectorStore(ABC):
    """Stores embedded chunks and finds the nearest ones to a query vector."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, MetadataValue]],
    ) -> None:
        """Insert or replace records by id. Blocking."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """Return up to `top_k` nearest records, best first. Blocking."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove records by id; unknown ids are ignored. Blocking."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored records."""

    @abstractmethod
    def reset(self) -> None:
        """Remove every record (e.g. when the embedding model changes)."""
