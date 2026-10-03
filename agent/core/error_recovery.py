import time


class ErrorRecoveryManager:
    '''Classify and bound retries for transient AI/provider errors.'''

    MAX_RETRIES = 2

    RETRY_DELAYS = (
        1,
        2,
    )

    RETRYABLE_MARKERS = (
        "500",
        "502",
        "503",
        "504",
        "BAD GATEWAY",
        "SERVICE UNAVAILABLE",
        "TEMPORARY",
        "TIMED OUT",
        "TIMEOUT",
        "CONNECTION RESET",
        "CONNECTION ABORTED",
        "CONNECTION ERROR",
        "NETWORK ERROR",
        "UNAVAILABLE",
    )

    NON_RETRYABLE_MARKERS = (
        "401",
        "403",
        "404",
        "INVALID API KEY",
        "API_KEY IS NOT SET",
        "INVALID JWT",
        "PERMISSION DENIED",
        "AUTHENTICATION",
        "AUTHORIZATION",
    )

    def should_retry(
        self,
        error_message: str,
        attempt: int,
    ) -> bool:
        '''Return True only for transient errors within retry budget.'''

        if attempt >= self.MAX_RETRIES:
            return False

        text = str(error_message).upper()

        if any(
            marker in text
            for marker in self.NON_RETRYABLE_MARKERS
        ):
            return False

        return any(
            marker in text
            for marker in self.RETRYABLE_MARKERS
        )

    def retry_delay(
        self,
        attempt: int,
    ) -> int:
        '''Return the bounded delay for a 1-based retry number.'''

        index = max(1, attempt) - 1
        return self.RETRY_DELAYS[
            min(index, len(self.RETRY_DELAYS) - 1)
        ]

    def sleep(
        self,
        delay: int,
    ) -> None:
        '''Sleep before a retry; isolated for easy testing.'''

        time.sleep(delay)
