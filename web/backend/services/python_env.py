import asyncio
import json
import os
import shutil
import subprocess

from schemas import EnvironmentInfo

from services.settings_manager import SettingsManager


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


class PythonEnvManager:
    def __init__(self, settings_manager: SettingsManager):
        self._settings_manager = settings_manager

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

    def resolve_command(
        self, env_id: str, custom_python_path: str | None = None
    ) -> list[str]:
        env_type, env_name = env_id.split(":", 1)
        if env_type == "pixi":
            pixi_bin = _find_pixi() or "pixi"
            return [pixi_bin, "run", "--environment", env_name, "python"]
        elif env_type == "conda":
            conda_bin = _find_conda() or "conda"
            return [conda_bin, "run", "-n", env_name, "--no-banner", "python"]
        elif env_type == "custom":
            if not custom_python_path:
                return ["python"]
            return [custom_python_path]
        return ["python"]

    def resolve_default_command(self) -> list[str]:
        default_env = self._settings_manager.get_default_env()
        if default_env:
            custom_path = self._settings_manager.get_custom_python_path()
            return self.resolve_command(default_env, custom_path)
        return ["python"]

    async def health_check(
        self, env_id: str, custom_python_path: str | None = None
    ) -> dict:
        cmd = self.resolve_command(env_id, custom_python_path)

        python_available = False
        python_version = None
        torch_available = False
        torch_version = None
        error = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                "-c",
                "print('ok')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutExpired:
                proc.kill()
                await proc.wait()
                error = "Python executable timed out"
            else:
                if proc.returncode == 0:
                    python_available = True
                else:
                    error = (
                        stderr.decode(errors="replace").strip()
                        or f"exit code {proc.returncode}"
                    )
        except FileNotFoundError:
            error = "Python executable not found"
        except Exception as e:
            error = str(e)

        if python_available:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    "-c",
                    "import sys; print(sys.version)",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                python_version = stdout.decode(errors="replace").strip()
            except Exception:
                pass

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    "-c",
                    "import torch; print(torch.__version__)",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode == 0:
                    torch_available = True
                    torch_version = stdout.decode(errors="replace").strip()
            except Exception:
                pass

        return {
            "env_id": env_id,
            "python_available": python_available,
            "python_version": python_version,
            "torch_available": torch_available,
            "torch_version": torch_version,
            "error": error,
        }

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
