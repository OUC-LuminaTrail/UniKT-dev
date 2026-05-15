# Python Environment Module Refactoring

## Goal

Replace the `EnvironmentResolver` class with a unified `PythonEnvManager` that provides environment discovery, command resolution, health checking, and default-environment management. Extend preprocess tasks to support per-task environment selection.

## Current State

- `services/environment_resolver.py` — `EnvironmentResolver` with `discover()` and `resolve_command()`; scans pixi, conda, and custom paths
- `services/process_manager.py` — Owns its own `EnvironmentResolver` instance; supports `env_id` + `custom_python_path` per training task
- `services/preprocess_manager.py` — Receives resolver + settings_manager via constructor; resolves base command from default env only via `_resolve_base_cmd()`; no per-task env selection
- `services/settings_manager.py` — Stores `default_env_id` and `custom_python_path` in `settings.json`; `set_default_env()` also sets `setup_completed`
- `routers/settings_api.py` — GET/POST `/api/settings/default-env`
- `routers/environments.py` — GET `/api/environments` creates a new `EnvironmentResolver()` per request
- `schemas.py` — `EnvironmentInfo(id, type, name, display_name, python_path)`
- `dependencies.py` — Singletons for managers; no singleton for `EnvironmentResolver`
- `main.py` — Creates `EnvironmentResolver` in lifespan, passes it to `PreprocessManager`
- Frontend `PreprocessView.vue` — No environment selector; always uses default env
- Frontend `TaskLaunch.vue` / `SelectionStep.vue` — Has environment selector
- Frontend `SettingsView.vue` — Has default env selector

## Design

### 1. New Module: `services/python_env.py`

Create `PythonEnvManager` to replace `EnvironmentResolver`. It consolidates all Python environment logic into a single class.

```python
class PythonEnvManager:
    def __init__(self, settings_manager: SettingsManager):
        self._settings_manager = settings_manager

    def discover(self) -> list[EnvironmentInfo]:
        # Same logic as current EnvironmentResolver.discover()
        # Scans pixi, conda, adds custom:0 entry

    def resolve_command(self, env_id: str, custom_python_path: str | None = None) -> list[str]:
        # Same logic as current EnvironmentResolver.resolve_command()

    def resolve_default_command(self) -> list[str]:
        """Resolve command using the default env from settings."""
        default_env = self._settings_manager.get_default_env()
        if default_env:
            custom_path = self._settings_manager.get_custom_python_path()
            return self.resolve_command(default_env, custom_path)
        return ["python"]

    async def health_check(self, env_id: str, custom_python_path: str | None = None) -> EnvHealthResult:
        """Run health checks on a given environment."""
        cmd = self.resolve_command(env_id, custom_python_path)
        # 1. Check executable exists and runs
        # 2. Get Python version
        # 3. Try importing torch, report version
        return EnvHealthResult(...)
```

The `health_check()` method runs three checks sequentially via `asyncio.create_subprocess_exec`:

1. **Executable check**: `python -c "print('ok')"` — verifies the Python binary runs
2. **Version check**: `python -c "import sys; print(sys.version)"` — returns version string
3. **Torch check**: `python -c "import torch; print(torch.__version__)"` — returns torch version or import error

Each check has a 10-second timeout. Results are aggregated into `EnvHealthResult`.

### 2. New Schema: `EnvHealthResult`

In `schemas.py`:

```python
class EnvHealthResult(BaseModel):
    env_id: str
    python_available: bool
    python_version: str | None = None
    torch_available: bool
    torch_version: str | None = None
    error: str | None = None
```

### 3. Delete `services/environment_resolver.py`

All logic moves into `PythonEnvManager`. The old file is deleted.

### 4. Update `dependencies.py`

Add a singleton for `PythonEnvManager`:

```python
python_env_manager: PythonEnvManager | None = None

def get_python_env_manager() -> PythonEnvManager:
    assert python_env_manager is not None
    return python_env_manager
```

### 5. Update `main.py`

In `lifespan()`, create `PythonEnvManager` instead of `EnvironmentResolver`:

```python
deps.python_env_manager = PythonEnvManager(settings_manager=deps.settings_manager)
```

Pass it to both `ProcessManager` and `PreprocessManager`.

### 6. Update `ProcessManager`

- Constructor receives `PythonEnvManager` instead of creating its own `EnvironmentResolver`
- All `self._resolver.resolve_command(...)` calls become `self._env_manager.resolve_command(...)`
- No other logic changes

### 7. Update `PreprocessManager`

- Constructor receives `PythonEnvManager` (already receives resolver; rename parameter)
- `start()` method gains optional `env_id` and `custom_python_path` parameters
- `_resolve_base_cmd()` is removed; replaced by inline logic in `_build_command()`:
  - If `env_id` is provided: `self._env_manager.resolve_command(env_id, custom_python_path)`
  - If not provided: `self._env_manager.resolve_default_command()`
- `PreprocessTask` gains `env_id` and `custom_python_path` fields

### 8. Update `routers/environments.py`

- Use `get_python_env_manager` dependency instead of creating `EnvironmentResolver()` per request
- Add health check endpoint:

```python
@router.post("/health-check")
async def health_check(
    body: EnvHealthCheckRequest,
    mgr: PythonEnvManager = Depends(get_python_env_manager),
):
    result = await mgr.health_check(body.env_id, body.custom_python_path)
    return result
```

Where `EnvHealthCheckRequest` is:

```python
class EnvHealthCheckRequest(BaseModel):
    env_id: str
    custom_python_path: str | None = None
```

### 9. Update `routers/preprocess.py`

- `PreprocessStartRequest` gains optional fields:

```python
class PreprocessStartRequest(BaseModel):
    action: str
    dataset: str
    params: dict = {}
    env_id: str | None = None
    custom_python_path: str | None = None
```

- `start_preprocess()` passes `env_id` and `custom_python_path` to `pm.start()`

### 10. Frontend: Add Environment Selector to PreprocessView

- Add environment selector UI to `PreprocessView.vue`, similar to `SelectionStep.vue`'s selector
- Fetch environments and default env on mount
- When no explicit env is selected, show a "使用默认环境" option
- Pass `env_id` and `custom_python_path` in `startPreprocess()` request
- Update `previewCommand` to reflect the selected environment's command prefix

### 11. Frontend: Add Health Check API

In `api/environments.ts`:

```typescript
export interface EnvHealthResult {
  env_id: string
  python_available: boolean
  python_version: string | null
  torch_available: boolean
  torch_version: string | null
  error: string | null
}

export const healthCheckEnv = (data: { env_id: string; custom_python_path?: string | null }) =>
  api.post<EnvHealthResult>('/environments/health-check', data).then(r => r.data)
```

### 12. Frontend: Add Health Check Button to SettingsView

In the default environment selector section of `SettingsView.vue`, add a "检测环境" button that calls `healthCheckEnv()` and displays results inline (Python version, torch availability).

### 13. Frontend: Update Preprocess API Type

In `api/preprocess.ts`, update `PreprocessStartRequest`:

```typescript
export interface PreprocessStartRequest {
  action: string
  dataset: string
  params: Record<string, any>
  env_id?: string | null
  custom_python_path?: string | null
}
```

### 14. Default Environment Setting System (Preserved)

The existing default environment setting flow is fully preserved:

- `SettingsManager.set_default_env(env_id, custom_python_path)` — saves to `settings.json`
- `SettingsManager.get_default_env()` / `get_custom_python_path()` — reads from settings
- `POST /api/settings/default-env` — updates default env
- `GET /api/settings/default-env` — reads default env
- Frontend `SettingsView.vue` — default env selector UI

The new `PythonEnvManager.resolve_default_command()` uses this system. Both training tasks and preprocess tasks fall back to the default env when no explicit env is specified.

## Files Changed

| File | Action |
|------|--------|
| `services/python_env.py` | **CREATE** — New `PythonEnvManager` class |
| `services/environment_resolver.py` | **DELETE** |
| `services/process_manager.py` | **MODIFY** — Use `PythonEnvManager` instead of `EnvironmentResolver` |
| `services/preprocess_manager.py` | **MODIFY** — Accept `PythonEnvManager`, add `env_id` param to `start()` |
| `dependencies.py` | **MODIFY** — Add `python_env_manager` singleton |
| `main.py` | **MODIFY** — Create `PythonEnvManager`, pass to managers |
| `schemas.py` | **MODIFY** — Add `EnvHealthResult` and `EnvHealthCheckRequest` |
| `routers/environments.py` | **MODIFY** — Use singleton, add health check endpoint |
| `routers/preprocess.py` | **MODIFY** — Accept `env_id`/`custom_python_path` in request |
| `frontend/src/api/environments.ts` | **MODIFY** — Add health check types and API |
| `frontend/src/api/preprocess.ts` | **MODIFY** — Add `env_id`/`custom_python_path` to request |
| `frontend/src/views/PreprocessView.vue` | **MODIFY** — Add environment selector |
| `frontend/src/views/SettingsView.vue` | **MODIFY** — Add health check button |

## Out of Scope

- Caching environment discovery results (envs are re-discovered per request, same as current behavior)
- Adding new environment types (system Python detection, etc.)
- Modifying the `SettingsManager` persistence layer
- Changing how training tasks select environments (already works)
