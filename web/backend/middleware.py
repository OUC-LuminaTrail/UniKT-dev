import json

from starlette.middleware.base import BaseHTTPMiddleware


class MessageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        messages = getattr(request.state, "messages", None)
        if messages and response.status_code < 400:
            response.headers["X-Messages"] = json.dumps(messages)
        return response
