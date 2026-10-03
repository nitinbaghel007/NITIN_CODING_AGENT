from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the language model."""
        raise NotImplementedError