"""Load the project .env once, at first import of utils.core.

Centralizes dotenv loading so every entry point and module sees the same
environment, before any module reads os.environ. Existing process env vars
take precedence over .env (override=False).
"""

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Load the project .env into os.environ; existing vars take precedence."""
    load_dotenv(find_dotenv(usecwd=False), override=False)


load_env()
