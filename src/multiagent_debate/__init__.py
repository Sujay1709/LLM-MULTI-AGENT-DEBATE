"""Reusable components for multi-agent language-model debate experiments."""

from .clients import FakeClient, LiteLLMClient, LLMClient, make_client
from .engine import run_debate
from .models import DebateConfig, Example, Generation, Message

__all__ = [
    "DebateConfig",
    "Example",
    "FakeClient",
    "Generation",
    "LLMClient",
    "LiteLLMClient",
    "Message",
    "make_client",
    "run_debate",
]
