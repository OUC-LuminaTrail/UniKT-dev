import asyncio
from pathlib import Path

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
        self, log_path: str, websocket, check_alive=None, from_offset: int = 0
    ):
        path = Path(log_path)
        if not path.exists():
            await websocket.send_json({"type": "error", "content": "Log file not found"})
            await websocket.close()
            return

        with open(path, "rb") as f:
            file_size = f.seek(0, 2)

            start = 0
            if from_offset > 0 and from_offset <= file_size:
                start = from_offset

            offset = start
            while offset < file_size:
                f.seek(offset)
                read_size = min(INITIAL_CHUNK_SIZE + 4, file_size - offset + 4)
                raw = f.read(read_size)
                if not raw:
                    break

                boundary = _find_safe_boundary(raw, min(INITIAL_CHUNK_SIZE, len(raw)))
                if boundary == 0 and offset == start:
                    boundary = min(INITIAL_CHUNK_SIZE, len(raw))
                chunk = raw[:boundary]

                offset += boundary
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk.decode("utf-8", errors="replace")
                await websocket.send_json(
                    {"type": "data", "content": text, "offset": offset}
                )
                await asyncio.sleep(0)

            if not check_alive or not check_alive():
                await websocket.send_json({"type": "done", "final": True})
                await websocket.close()
                return

            while True:
                if check_alive and not check_alive():
                    await self._send_remaining(f, websocket, offset)
                    break

                f.seek(offset)
                new_data = f.read()
                if new_data:
                    offset = f.tell()
                    try:
                        text = new_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = new_data.decode("utf-8", errors="replace")
                    await websocket.send_json(
                        {"type": "data", "content": text, "offset": offset}
                    )
                else:
                    await asyncio.sleep(0.3)

                if check_alive and not check_alive():
                    await self._send_remaining(f, websocket, offset)
                    break

        await websocket.send_json({"type": "done", "final": True})
        await websocket.close()

    async def _send_remaining(self, f, websocket, last_sent_offset: int):
        f.seek(0, 2)
        file_size = f.tell()
        if file_size <= last_sent_offset:
            return
        f.seek(last_sent_offset)
        remaining = f.read()
        if remaining:
            offset = f.tell()
            try:
                text = remaining.decode("utf-8")
            except UnicodeDecodeError:
                text = remaining.decode("utf-8", errors="replace")
            await websocket.send_json(
                {"type": "data", "content": text, "offset": offset}
            )
