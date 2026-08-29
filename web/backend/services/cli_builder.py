"""Shared helper to build CLI flags from flat frontend params.

Both ProcessManager (train.py) and PreprocessManager (data_process.py) route
flat form params into dotted ``--node.field=value`` flags via cached schema
routes, skipping None and default-equal values. Preprocess also has flat flags
(--force/--extra); ``allow_flat_node`` toggles that.
"""


def is_default(value: object, default: object) -> bool:
    """Return True when *value* equals the schema default.

    Treats ``False`` as equivalent to a ``None`` default to compensate for the
    frontend's ``el-switch`` coercing ``null`` to ``false`` on optional boolean
    fields like ``compile_dynamic``.
    """
    if isinstance(value, list) and isinstance(default, list):
        return value == default
    if default is None and value is False:
        return True
    return value == default


def build_param_flags(
    params: dict,
    routes: dict[str, str],
    defaults: dict[str, object],
    *,
    allow_flat_node: bool = False,
) -> list[str]:
    """Build ``--node.field=value`` (or flat ``--field=value``) flags from params.

    Args:
        params: flat ``{field: value}`` from the frontend form.
        routes: ``{field: node}`` — node is the RunConfig node (``"data"`` /
            ``"general"`` / ...), or ``""`` for an explicit flat flag; keys
            missing from ``routes`` are skipped entirely.
        defaults: ``{field: default}`` — params equal to their default are skipped.
        allow_flat_node: if True, an empty-node entry yields a flat ``--field``
            flag (preprocess ``--force``/``--extra``); if False, flat entries
            are skipped (model params are always nested under a node).

    Returns:
        A list of CLI flag strings.
    """
    flags: list[str] = []
    for field, value in params.items():
        node = routes.get(field)
        if node is None:
            # Unrouted key — never a legal CLI target; skipping keeps clients
            # from injecting arbitrary flags such as --data_url/--config.
            continue
        is_flat = not node
        if is_flat and not allow_flat_node:
            continue
        if value is None:
            continue
        if is_default(value, defaults.get(field)):
            continue
        prefix = f"--{field}" if is_flat else f"--{node}.{field}"
        if isinstance(value, bool):
            flags.append(f"{prefix}={str(value).lower()}")
        elif isinstance(value, list):
            flags.append(f"{prefix}=[{','.join(str(v) for v in value)}]")
        else:
            flags.append(f"{prefix}={value}")
    return flags
