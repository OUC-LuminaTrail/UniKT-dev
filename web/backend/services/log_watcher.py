import asyncio
from pathlib import Path


class LogWatcher:
    async def stream_log(self, log_path: str, websocket, check_alive=None):
        path = Path(log_path)
        if not path.exists():
            await websocket.send_json({"type": "error", "content": "Log file not found"})
            await websocket.close()
            return

        with open(path, "r") as f:
            lines = f.readlines()
            recent = lines[-500:] if len(lines) > 500 else lines
            for line in recent:
                await websocket.send_json({"type": "data", "content": line})
                await asyncio.sleep(0)

        offset = f.tell()

        while True:
            if check_alive and not check_alive():
                await self._send_remaining(f, websocket)
                break

            f.seek(offset)
            new_data = f.read()
            if new_data:
                offset = f.tell()
                for line in new_data.splitlines(keepends=True):
                    await websocket.send_json({"type": "data", "content": line})
            else:
                await asyncio.sleep(0.5)

            if check_alive and not check_alive():
                await self._send_remaining(f, websocket)
                break

        await websocket.send_json({"type": "done"})
        await websocket.close()

    async def _send_remaining(self, f, websocket):
        f.seek(0, 2)
        pos = f.tell()
        f.seek(max(0, pos - 4096))
        remaining = f.read()
        if remaining:
            for line in remaining.splitlines(keepends=True):
                await websocket.send_json({"type": "data", "content": line})
