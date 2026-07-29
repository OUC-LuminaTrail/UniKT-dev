"""Application errors with stable i18n codes for the frontend.

Each user-facing error is raised as an :class:`AppError` carrying a stable
``code``. The fastapi_problem handler (registered in ``main.py``) renders it
into the RFC 9457 Problem ``type`` field; the frontend maps ``type`` to a
localized message via vue-i18n, falling back to ``detail`` when missing.
"""


class AppError(Exception):
    """User-facing error carrying a stable i18n code.

    Attributes:
        code: Stable snake_case identifier rendered into ``Problem.type``.
        status: HTTP status code.
        detail: Optional fallback message; defaults to the code.
    """

    def __init__(self, code: str, status: int = 400, detail: str | None = None):
        """Store the i18n code, HTTP status, and optional fallback detail."""
        self.code = code
        self.status = status
        self.detail = detail
        super().__init__(code)
