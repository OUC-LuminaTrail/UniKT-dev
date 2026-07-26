"""File-backed log reading and WebSocket streaming.

Training/preprocess output is appended to a per-task ``.log`` file; this module
reads it back by byte offset. Keeps the WebSocket protocol (``data``/``done``/
``offset``/``error``) the frontend already expects, replacing the old
DB-per-chunk storage.
"""

import asyncio
from pathlib import Path

from services.log_watcher import _find_safe_boundary

CHUNK_SIZE = 65536
# Hard cap on a single read so a huge limit cannot pull gigabytes into memory.
MAX_READ_BYTES = 16 * 1024 * 1024


def read_log_text(path: Path, offset: int = 0, limit: int = 500) -> dict:
    """Read a log file from ``offset`` as decoded text.

    Args:
        path: Log file path.
        offset: Byte offset to start at.
        limit: Maximum number of 64KB chunks to read.

    Returns:
        ``{"content": str, "total_bytes": int}``; empty content if the file is
        missing or already fully read.
    """
    if not path.is_file():
        return {"content": "", "total_bytes": 0}
    size = path.stat().st_size
    to_read = min(size - offset, CHUNK_SIZE * limit, MAX_READ_BYTES)
    if to_read <= 0:
        return {"content": "", "total_bytes": size}
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(to_read)
    boundary = _find_safe_boundary(data, len(data))
    try:
        text = data[:boundary].decode("utf-8")
    except UnicodeDecodeError:
        text = data[:boundary].decode("utf-8", errors="replace")
    return {"content": text, "total_bytes": size}


async def stream_log(
    path: Path,
    websocket,
    check_alive=None,
    from_offset: int = 0,
):
    """Stream a log file over a WebSocket, polling while the source is alive.

    Sends ``data`` messages with decoded text and the next byte offset, then a
    final ``done`` message when the source process has exited.

    Args:
        path: Log file path.
        websocket: WebSocket connection to send messages over.
        check_alive: Optional callable returning whether the source is running.
        from_offset: Starting byte offset.
    """
    offset = from_offset

    def read_one(off: int) -> tuple[bytes, int]:
        if not path.is_file():
            return b"", off
        with open(path, "rb") as f:
            f.seek(off)
            data = f.read(CHUNK_SIZE)
        if not data:
            return b"", off
        boundary = _find_safe_boundary(data, len(data))
        end = boundary if boundary > 0 else len(data)
        return data[:end], off + end

    async def send(data: bytes, next_offset: int) -> None:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        await websocket.send_json(
            {"type": "data", "content": text, "offset": next_offset}
        )

    while True:
        data, offset = await asyncio.to_thread(read_one, offset)
        if not data:
            break
        await send(data, offset)

    if not check_alive or not await asyncio.to_thread(check_alive):
        await websocket.send_json({"type": "done", "final": True})
        await websocket.close()
        return

    while True:
        alive = await asyncio.to_thread(check_alive)
        while True:
            data, offset = await asyncio.to_thread(read_one, offset)
            if not data:
                break
            await send(data, offset)
        if not alive:
            break
        await asyncio.sleep(0.3)

    await websocket.send_json({"type": "done", "final": True})
    await websocket.close()
