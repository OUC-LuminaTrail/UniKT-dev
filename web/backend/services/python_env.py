"""Python environment manager — discovery and command resolution.

Discovers pixi and conda environments, resolves Python interpreter commands
for a given environment, and provides async health checks (Python + PyTorch
availability).
"""

import asyncio
import contextlib
import json
import os
import shutil
import subprocess

from errors import AppError
from schemas import EnvironmentInfo

from services.settings_manager import SettingsManager


def _find_pixi() -> str | None:
    """Locate the pixi binary on the system.

    Checks PATH first, then common installation locations.

    Returns:
        The pixi binary path, or None if not found.
    """
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
    """Locate the conda binary on PATH.

    Returns:
        The conda binary path, or None if not found.
    """
    return shutil.which("conda")


class EnvironmentNotConfigured(AppError):
    """No Python environment is available for subprocess launch.

    Raised (never silently worked around) when neither an explicit ``env_id``
    nor a wizard-configured default exists, or a custom env has no path. The
    web backend's own interpreter lacks torch, so falling back to a bare
    ``python`` would just defer the failure into an opaque subprocess error —
    callers surface this to the user as a setup prompt instead.
    """

    def __init__(self, code: str = "env_not_configured") -> None:
        """Initialize with a stable i18n code (default env_not_configured)."""
        super().__init__(code, status=400)


class PythonEnvManager:
    """Discovers Python environments and resolves commands for subprocess launch.

    Supports pixi, conda, and custom-path Python environments, including
    async health checks that verify Python and PyTorch availability.
    """

    def __init__(self, settings_manager: SettingsManager):
        """Initialize the environment manager.

        Args:
            settings_manager: SettingsManager used for default environment lookups.
        """
        self._settings_manager = settings_manager

    def discover(self) -> list[EnvironmentInfo]:
        """Discover all available Python environments.

        Scans pixi environments, conda environments, and adds a custom
        Python path option.

        Returns:
            A list of EnvironmentInfo objects.
        """
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
                display_name="Custom Python path",
            )
        )
        return envs

    def resolve_command(
        self, env_id: str | None = None, custom_python_path: str | None = None
    ) -> list[str]:
        """Resolve the Python interpreter command for an environment.

        Args:
            env_id: Environment identifier (e.g. ``pixi:default``,
                ``conda:base``, ``custom:0``). If None, falls back to the
                wizard-configured default environment.
            custom_python_path: Custom interpreter path (for ``custom`` type).
                When ``env_id`` is None, read from settings alongside the
                default env.

        Returns:
            A command list suitable for subprocess execution.

        Raises:
            EnvironmentNotConfigured: If no ``env_id`` is given and no default
                environment is configured, or a custom env has no path. Never
                silently falls back to a bare ``python``.
        """
        if env_id is None:
            env_id = self._settings_manager.get_default_env()
            if env_id is None:
                raise EnvironmentNotConfigured()
            custom_python_path = self._settings_manager.get_custom_python_path()

        parts = env_id.split(":", 1)
        env_type = parts[0]
        env_name = parts[1] if len(parts) > 1 else ""
        if env_type == "pixi":
            pixi_bin = _find_pixi() or "pixi"
            return [pixi_bin, "run", "--environment", env_name, "python"]
        if env_type == "conda":
            conda_bin = _find_conda() or "conda"
            return [conda_bin, "run", "-n", env_name, "--no-banner", "python"]
        if env_type == "custom":
            if not custom_python_path:
                raise EnvironmentNotConfigured("custom_env_no_path")
            return [custom_python_path]
        raise ValueError(f"Unknown environment type: {env_type!r} (env_id={env_id})")

    async def health_check(
        self, env_id: str, custom_python_path: str | None = None
    ) -> dict:
        """Run an asynchronous health check on a Python environment.

        Checks Python availability, version, and PyTorch availability.

        Args:
            env_id: Environment identifier.
            custom_python_path: Optional custom Python interpreter path.

        Returns:
            A dict with ``env_id``, ``python_available``, ``python_version``,
            ``torch_available``, ``torch_version``, and ``error``.
        """
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
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
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
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                    raise
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
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                    raise
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
        """Scan pixi environments via ``pixi info --json``.

        Args:
            pixi_bin: Path to the pixi binary.

        Returns:
            A list of EnvironmentInfo objects for pixi environments.
        """
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
        """Scan conda environments via ``conda env list --json``.

        Args:
            conda_bin: Path to the conda binary.

        Returns:
            A list of EnvironmentInfo objects for conda environments.
        """
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
