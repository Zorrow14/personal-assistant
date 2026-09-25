"""Select an LLMClient implementation from settings."""

from jarvis.config import Settings
from jarvis.core.interfaces import LLMClient, LLMError

SUPPORTED_PROVIDERS = ("gemini",)


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the LLMClient named by `settings.llm_provider`.

    Provider modules are imported lazily so unused SDKs are never loaded.

    Raises:
        LLMError: If the provider is unknown or misconfigured.
    """
    if settings.llm_provider == "gemini":
        from jarvis.llm.gemini_client import GeminiClient

        return GeminiClient.from_settings(settings)
    raise LLMError(
        f"unknown llm_provider {settings.llm_provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
