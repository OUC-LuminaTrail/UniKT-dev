"""Log streaming watcher for WebSocket-based log delivery.

Reads log chunks from the database for a given source/source_id and streams
them over a WebSocket connection with byte-offset tracking and safe UTF-8
boundary detection.
"""

import asyncio

from database import SessionLocal
from models import LogChunk
from sqlalchemy import asc

INITIAL_CHUNK_SIZE = 65536


def _find_safe_boundary(data: bytes, intended_end: int) -> int:
    """Find a safe UTF-8 character boundary near the intended end.

    Walks backward from ``intended_end`` to avoid splitting a multi-byte
    UTF-8 character.

    Args:
        data: The raw bytes to search in.
        intended_end: The desired cut position.

    Returns:
        A safe byte offset not exceeding ``intended_end`` that falls on a
        character boundary.
    """
    i = intended_end
    while i > 0:
        b = data[i - 1]
        if (b & 0b11_000000) != 0b10_000000:
            if (b & 0b1111_0000) == 0b1111_0000:
                needed = 4
            elif (b & 0b1110_0000) == 0b1110_0000:
                needed = 3
            elif (b & 0b1100_0000) == 0b1100_0000:
                needed = 2
            else:
                needed = 1
            char_end = i - 1 + needed
            if char_end <= intended_end:
                return char_end
            return i - 1
        i -= 1
    return 0


class LogWatcher:
    """Streams log chunks from the database over a WebSocket.

    Reads stored log chunks for a given source and source_id and sends
    them as JSON messages over the WebSocket, polling for new data when
    the process is still alive.
    """

    async def stream_log(
        self,
        source: str,
        source_id: int,
        websocket,
        check_alive=None,
        from_offset: int = 0,
    ):
        """Stream log chunks over a WebSocket connection.

        Sends ``data`` JSON messages with decoded text and the next byte offset,
        followed by a final ``done`` message.

        Args:
            source: The log source type (e.g. "task" or "preprocess").
            source_id: The source entity identifier.
            websocket: The WebSocket connection to send messages to.
            check_alive: Optional callable returning whether the source process
                is still running.
            from_offset: Starting byte offset for reading log chunks.
        """
        offset = from_offset
        chunks = await asyncio.to_thread(self._read_chunks, source, source_id, offset)
        for raw_data, chunk_offset in chunks:
            boundary = _find_safe_boundary(
                raw_data, min(INITIAL_CHUNK_SIZE, len(raw_data))
            )
            if boundary > 0:
                chunk = raw_data[:boundary]
                offset = chunk_offset + boundary
            else:
                chunk = raw_data
                offset = chunk_offset + len(raw_data)
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                text = chunk.decode("utf-8", errors="replace")
            await websocket.send_json(
                {"type": "data", "content": text, "offset": offset}
            )
            await asyncio.sleep(0)

        if not check_alive or not await asyncio.to_thread(check_alive):
            await websocket.send_json({"type": "done", "final": True})
            await websocket.close()
            return

        while True:
            if check_alive and not await asyncio.to_thread(check_alive):
                remaining = await asyncio.to_thread(
                    self._read_chunks, source, source_id, offset
                )
                for raw_data, chunk_offset in remaining:
                    try:
                        text = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_data.decode("utf-8", errors="replace")
                    offset = chunk_offset + len(raw_data)
                    await websocket.send_json(
                        {"type": "data", "content": text, "offset": offset}
                    )
                break

            new_chunks = await asyncio.to_thread(
                self._read_chunks, source, source_id, offset
            )
            if new_chunks:
                for raw_data, chunk_offset in new_chunks:
                    try:
                        text = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_data.decode("utf-8", errors="replace")
                    offset = chunk_offset + len(raw_data)
                    await websocket.send_json(
                        {"type": "data", "content": text, "offset": offset}
                    )
            else:
                await asyncio.sleep(0.3)

            if check_alive and not await asyncio.to_thread(check_alive):
                remaining = await asyncio.to_thread(
                    self._read_chunks, source, source_id, offset
                )
                for raw_data, chunk_offset in remaining:
                    try:
                        text = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_data.decode("utf-8", errors="replace")
                    offset = chunk_offset + len(raw_data)
                    await websocket.send_json(
                        {"type": "data", "content": text, "offset": offset}
                    )
                break

        await websocket.send_json({"type": "done", "final": True})
        await websocket.close()

    def _read_chunks(
        self, source: str, source_id: int, from_offset: int
    ) -> list[tuple[bytes, int]]:
        """Read log chunks from the database starting at the given offset.

        Args:
            source: The log source type.
            source_id: The source entity identifier.
            from_offset: Minimum byte offset to start reading from.

        Returns:
            A list of ``(raw_data, byte_offset)`` tuples ordered by offset.
        """
        with SessionLocal() as session:
            rows = (
                session.query(LogChunk.raw_data, LogChunk.byte_offset)
                .filter(
                    LogChunk.source == source,
                    LogChunk.source_id == source_id,
                    LogChunk.byte_offset >= from_offset,
                )
                .order_by(asc(LogChunk.byte_offset))
                .all()
            )
            return [(row[0], row[1]) for row in rows]
