import asyncio

from database import SessionLocal
from models import LogChunk
from sqlalchemy import asc

INITIAL_CHUNK_SIZE = 65536


def _find_safe_boundary(data: bytes, intended_end: int) -> int:
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
    async def stream_log(
        self,
        source: str,
        source_id: int,
        websocket,
        check_alive=None,
        from_offset: int = 0,
    ):
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
