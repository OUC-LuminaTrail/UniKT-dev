import json
import os
import shutil
import subprocess

from schemas import EnvironmentInfo


def _find_pixi() -> str | None:
    path = shutil.which("pixi")
    if path:
        return path
    candidates = [
        os.path.expanduser("~/.pixi/bin/pixi"),
        os.path.expanduser("~/.local/share/mise/installs/pixi/current/pixi"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _find_conda() -> str | None:
    return shutil.which("conda")


class EnvironmentResolver:
    def discover(self) -> list[EnvironmentInfo]:
        envs: list[EnvironmentInfo] = []

        pixi_bin = _find_pixi()
        if pixi_bin:
            envs.extend(self._scan_pixi(pixi_bin))

        conda_bin = _find_conda()
        if conda_bin:
            envs.extend(self._scan_conda(conda_bin))

        envs.append(
            EnvironmentInfo(
                id="custom:0",
                type="custom",
                name="custom",
                display_name="自定义 Python 路径",
            )
        )
        return envs

    def _scan_pixi(self, pixi_bin: str) -> list[EnvironmentInfo]:
        envs = []
        try:
            result = subprocess.run(
                [pixi_bin, "info", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            data = json.loads(result.stdout)

            env_list = data.get("environments_info") or data.get("environments") or []
            for env_info in env_list:
                name = env_info.get("name", "default")
                features = env_info.get("features", [])
                display = f"Pixi - {name}"
                if features:
                    display += f" ({', '.join(features)})"
                envs.append(
                    EnvironmentInfo(
                        id=f"pixi:{name}",
                        type="pixi",
                        name=name,
                        display_name=display,
                    )
                )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return envs

    def _scan_conda(self, conda_bin: str) -> list[EnvironmentInfo]:
        envs = []
        try:
            result = subprocess.run(
                [conda_bin, "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            data = json.loads(result.stdout)
            for env_path in data.get("envs", []):
                name = env_path.rsplit("/", 1)[-1]
                if not name:
                    continue
                envs.append(
                    EnvironmentInfo(
                        id=f"conda:{name}",
                        type="conda",
                        name=name,
                        display_name=f"Conda - {name}",
                    )
                )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return envs

    def resolve_command(self, env_id: str, custom_python_path: str | None = None) -> list[str]:
        env_type, env_name = env_id.split(":", 1)
        if env_type == "pixi":
            pixi_bin = _find_pixi() or "pixi"
            return [pixi_bin, "run", "--environment", env_name, "python"]
        elif env_type == "conda":
            return ["conda", "run", "-n", env_name, "--no-banner", "python"]
        elif env_type == "custom":
            if not custom_python_path:
                return ["python"]
            return [custom_python_path]
        return ["python"]
