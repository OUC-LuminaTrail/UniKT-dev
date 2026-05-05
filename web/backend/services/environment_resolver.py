import json
import shutil
import subprocess

from schemas import EnvironmentInfo


class EnvironmentResolver:
    def discover(self) -> list[EnvironmentInfo]:
        envs: list[EnvironmentInfo] = []
        if shutil.which("pixi"):
            envs.extend(self._scan_pixi())
        if shutil.which("conda"):
            envs.extend(self._scan_conda())
        envs.append(
            EnvironmentInfo(
                id="custom:0",
                type="custom",
                name="custom",
                display_name="自定义 Python 路径",
            )
        )
        return envs

    def _scan_pixi(self) -> list[EnvironmentInfo]:
        envs = []
        try:
            result = subprocess.run(
                ["pixi", "info", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            data = json.loads(result.stdout)
            for env_info in data.get("environments", []):
                name = env_info.get("name", "default")
                display = f"Pixi - {name}"
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

    def _scan_conda(self) -> list[EnvironmentInfo]:
        envs = []
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
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
            return ["pixi", "run", "--environment", env_name, "python"]
        elif env_type == "conda":
            return ["conda", "run", "-n", env_name, "--no-banner", "python"]
        elif env_type == "custom":
            if not custom_python_path:
                return ["python"]
            return [custom_python_path]
        return ["python"]
