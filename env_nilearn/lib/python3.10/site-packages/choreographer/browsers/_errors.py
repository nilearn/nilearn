class BrowserClosedError(RuntimeError):
    """An error for when the browser is closed accidentally (during access)."""


class BrowserFailedError(RuntimeError):
    """An error for when the browser fails to launch."""


class BrowserDepsError(BrowserFailedError):
    """An error for when the browser is closed because of missing libs."""

    def __init__(self) -> None:
        msg = (
            "It seems like you are running a slim version of your "
            "operating system and are missing some common dependencies. "
            "The following command should install the required "
            "dependencies on most systems:\n"
            "\n"
            "$ sudo apt update && sudo apt-get install libnss3 "
            "libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 "
            "libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 "
            "libcairo2 libasound2\n"
            "\n"
            "If you have already run the above command and are still "
            "seeing this error, or the above command fails, consult the "
            "Kaleido documentation for operating system to install "
            "chromium dependencies.\n"
            "\n"
            "For support, run the command `choreo_diagnose` and create "
            "an issue with its output."
        )
        super().__init__(msg)


# BrowserDeps being a more specific type of Failure.
# And Closed not necessarily being related (you can intentionally closed,
# and something else can error because its closed.)
