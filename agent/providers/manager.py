from agent.providers.base import LLMProvider


class ProviderManager:
    """Manage and select LLM providers for Nitin Coding Agent."""

    FAILOVER_ORDER = (
        "openrouter",
        "gemini",
    )

    UNAVAILABLE_STATUSES = {
        "rate_limited",
        "error",
        "disabled",
    }

    def __init__(
        self,
        providers: dict[str, LLMProvider] | None = None,
        default_provider: str = "openrouter",
    ):
        self.providers = providers or {}
        self.default_provider = default_provider

        self.provider_status: dict[str, str] = {}

        for name in self.providers:
            self.provider_status[name] = "registered"

    def register(
        self,
        name: str,
        provider: LLMProvider,
    ) -> None:
        if not name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        if not isinstance(
            provider,
            LLMProvider,
        ):
            raise TypeError(
                "Provider must implement LLMProvider."
            )

        self.providers[name] = provider
        self.provider_status[name] = "registered"

    def get(
        self,
        name: str | None = None,
    ) -> LLMProvider:
        provider_name = (
            name
            if name is not None
            else self.default_provider
        )

        if provider_name not in self.providers:
            raise RuntimeError(
                f"Provider not available: "
                f"{provider_name}"
            )

        return self.providers[
            provider_name
        ]

    def available(self) -> list[str]:
        return sorted(
            self.providers.keys()
        )

    def set_default(
        self,
        name: str,
    ) -> None:
        if name not in self.providers:
            raise RuntimeError(
                f"Provider not available: "
                f"{name}"
            )

        self.default_provider = name

    def set_status(
        self,
        name: str,
        status: str,
    ) -> None:
        if name not in self.providers:
            raise RuntimeError(
                f"Provider not available: "
                f"{name}"
            )

        allowed_statuses = {
            "registered",
            "available",
            "rate_limited",
            "error",
            "disabled",
        }

        if status not in allowed_statuses:
            raise ValueError(
                f"Invalid provider status: {status}"
            )

        self.provider_status[name] = status

    def get_status(
        self,
        name: str,
    ) -> str:
        if name not in self.providers:
            raise RuntimeError(
                f"Provider not available: {name}"
            )

        return self.provider_status.get(
            name,
            "registered",
        )

    def failover(
        self,
        failed_provider: str,
    ) -> str | None:
        """Switch to the next usable registered provider.

        Only providers that are not rate-limited, errored, or disabled
        are eligible. This keeps failover inside the configured free
        provider pool and never introduces a paid provider.
        """

        if failed_provider in self.providers:
            self.provider_status[failed_provider] = (
                "rate_limited"
                if self.get_status(failed_provider) == "rate_limited"
                else self.get_status(failed_provider)
            )

        candidates = [
            name
            for name in self.providers
            if name != failed_provider
            and self.get_status(name)
            not in self.UNAVAILABLE_STATUSES
        ]

        ordered_candidates = sorted(
            candidates,
            key=self._failover_sort_key,
        )

        if not ordered_candidates:
            return None

        next_provider = ordered_candidates[0]
        self.default_provider = next_provider
        return next_provider

    def _failover_sort_key(
        self,
        name: str,
    ) -> tuple[int, str]:
        try:
            priority = self.FAILOVER_ORDER.index(name)
        except ValueError:
            priority = len(self.FAILOVER_ORDER)

        return priority, name

    def status(self) -> dict:
        return {
            "default_provider": self.default_provider,
            "available_providers": self.available(),
            "provider_status": dict(
                self.provider_status
            ),
        }
