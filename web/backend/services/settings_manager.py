import contextlib
import json
import os
import tempfile

from config import DATABASE_PATH

SETTINGS_PATH = DATABASE_PATH.parent / "settings.json"

_DEFAULT_SETTINGS = {
    "default_env_id": None,
    "custom_python_path": None,
    "setup_completed": False,
}


class SettingsManager:
    def __init__(self, path=None):
        self._path = path or SETTINGS_PATH

    def load(self) -> dict:
        if not self._path.is_file():
            return dict(_DEFAULT_SETTINGS)
        try:
            with open(self._path) as f:
                data = json.load(f)
            return {**_DEFAULT_SETTINGS, **data}
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULT_SETTINGS)

    def save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        merged = {**_DEFAULT_SETTINGS, **data}
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(self._path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def get_default_env(self) -> str | None:
        return self.load().get("default_env_id")

    def get_custom_python_path(self) -> str | None:
        return self.load().get("custom_python_path")

    def set_default_env(self, env_id: str, custom_python_path: str | None = None) -> None:
        self.save({
            "default_env_id": env_id,
            "custom_python_path": custom_python_path,
            "setup_completed": True,
        })

    def is_setup_completed(self) -> bool:
        return bool(self.load().get("setup_completed"))
