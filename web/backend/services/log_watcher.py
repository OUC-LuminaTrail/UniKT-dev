import asyncio
from pathlib import Path


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

            if from_offset > 0 and from_offset <= file_size:
                f.seek(from_offset)
                data = f.read()
            elif from_offset == 0:
                f.seek(0)
                data = f.read()
                if len(data) > 50000:
                    data = b"..." + data[-50000:]
            else:
                f.seek(0)
                data = f.read()

            offset = f.tell()

            if data:
                try:
                    text = data.decode("utf-8", errors="replace")
                except Exception:
                    text = data.decode("latin-1")
                await websocket.send_json({"type": "data", "content": text, "offset": offset})
                await asyncio.sleep(0)

            if not check_alive or not check_alive():
                await websocket.send_json({"type": "done", "final": True})
                await websocket.close()
                return

            while True:
                if check_alive and not check_alive():
                    await self._send_remaining(f, websocket)
                    break

                f.seek(offset)
                new_data = f.read()
                if new_data:
                    offset = f.tell()
                    try:
                        text = new_data.decode("utf-8", errors="replace")
                    except Exception:
                        text = data.decode("latin-1")
                    await websocket.send_json({"type": "data", "content": text, "offset": offset})
                else:
                    await asyncio.sleep(0.3)

                if check_alive and not check_alive():
                    await self._send_remaining(f, websocket)
                    break

        await websocket.send_json({"type": "done", "final": True})
        await websocket.close()

    async def _send_remaining(self, f, websocket):
        f.seek(0, 2)
        pos = f.tell()
        f.seek(max(0, pos - 8192))
        remaining = f.read()
        if remaining:
            try:
                text = remaining.decode("utf-8", errors="replace")
            except Exception:
                text = remaining.decode("latin-1")
            await websocket.send_json({"type": "data", "content": text, "offset": f.tell()})