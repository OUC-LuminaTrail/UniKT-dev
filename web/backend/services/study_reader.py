"""Read live trial progress from an Optuna study database (SQLite).

Used by the search detail view to surface trial counts, the current best value,
and a recent-trial table without importing Optuna — the web environment need not
have Optuna installed. Reads the standard RDBStorage schema written by Optuna
3.x+ and tolerates the minor version differences (string vs integer ``state``,
``trial_values`` presence).

Opens the database read-only (``mode=ro`` URI) so concurrent reads never contend
with the running search process's writes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Optuna stores TrialState as the enum name string in modern versions, but older
# releases used the integer value. Normalise both to the canonical name.
_STATE_INT_TO_STR = {
    0: "WAITING",
    1: "RUNNING",
    2: "COMPLETE",
    3: "PRUNED",
    4: "FAIL",
}


def _norm_state(value: object) -> str:
    """Normalise a ``trials.state`` cell to a canonical state name."""
    if isinstance(value, int):
        return _STATE_INT_TO_STR.get(value, str(value))
    return str(value)


def _convert_param_value(raw: object, distribution_json: object) -> object:
    """Restore a param's Python value from its stored float + distribution.

    Optuna's RDBStorage stores every ``param_value`` as a float: numeric params
    keep their value (ints as ``3.0``), but categorical params store the
    ``choices`` index. The paired ``distribution_json`` recovers the true type.
    """
    if not isinstance(raw, (int, float)):
        return raw
    if not isinstance(distribution_json, str):
        return raw
    try:
        dist = json.loads(distribution_json)
    except json.JSONDecodeError:
        return raw
    name = dist.get("name", "")
    attrs = (
        dist.get("attributes", {}) if isinstance(dist.get("attributes"), dict) else {}
    )
    if name == "IntDistribution":
        return int(raw)
    if name == "CategoricalDistribution":
        choices = attrs.get("choices") or []
        idx = int(raw)
        return choices[idx] if 0 <= idx < len(choices) else raw
    return raw  # FloatDistribution (or unknown) keeps the stored value


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _state_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_state, n in conn.execute(
        "SELECT state, COUNT(*) FROM trials GROUP BY state"
    ).fetchall():
        counts[_norm_state(raw_state)] = counts.get(_norm_state(raw_state), 0) + n
    return counts


def _best_trial(
    conn: sqlite3.Connection, direction: str, has_values: bool
) -> dict | None:
    """Return the best complete trial (number + value + params), or None."""
    if not has_values:
        return None
    rows = conn.execute(
        """
        SELECT t.trial_id, t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.state IN ('COMPLETE', 2) AND tv.value IS NOT NULL
        """
    ).fetchall()
    if not rows:
        return None
    picker = max if direction == "maximize" else min
    trial_id, number, value = picker(rows, key=lambda r: float(r[2]))
    return {
        "number": number,
        "value": float(value),
        "params": _trial_params(conn, trial_id),
    }


def _trial_params(conn: sqlite3.Connection, trial_id: int) -> dict:
    rows = conn.execute(
        "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?",
        (trial_id,),
    ).fetchall()
    return {name: _convert_param_value(value, dist) for name, value, dist in rows}


def _recent_trials(
    conn: sqlite3.Connection, has_values: bool, limit: int
) -> list[dict]:
    value_select = "tv.value" if has_values else "NULL AS value"
    join = "LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id" if has_values else ""
    rows = conn.execute(
        f"""
        SELECT t.trial_id, t.number, t.state, t.datetime_start,
               t.datetime_complete, {value_select}
        FROM trials t {join}
        ORDER BY t.number DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    trials: list[dict] = []
    for trial_id, number, raw_state, start, complete, value in rows:
        trials.append(
            {
                "number": number,
                "state": _norm_state(raw_state),
                "value": None if value is None else float(value),
                "params": _trial_params(conn, trial_id),
                "datetime_start": start,
                "datetime_complete": complete,
            }
        )
    return trials


def read_study(
    study_db_path: Path, direction: str = "maximize", limit: int = 20
) -> dict | None:
    """Read a trial-progress summary from an Optuna ``study.db``.

    Args:
        study_db_path: Path to the ``study.db`` file.
        direction: ``"maximize"`` or ``"minimize"`` — selects the best trial.
        limit: Maximum number of recent trials to return.

    Returns:
        A summary dict, or None when the DB is absent, not yet a valid Optuna
        study, or unreadable (e.g. the search has not created it yet). Callers
        render an empty state for None.
    """
    if not study_db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{study_db_path}?mode=ro", uri=True)
        try:
            tables = _existing_tables(conn)
            if "trials" not in tables:
                return None
            has_values = "trial_values" in tables
            counts = _state_counts(conn)
            total = sum(counts.values())
            return {
                "total": total,
                "completed": counts.get("COMPLETE", 0),
                "running": counts.get("RUNNING", 0),
                "pruned": counts.get("PRUNED", 0),
                "failed": counts.get("FAIL", 0),
                "direction": direction,
                "best_trial": _best_trial(conn, direction, has_values),
                "trials": _recent_trials(conn, has_values, limit=limit),
            }
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        logger.warning("study.db unreadable: %s", study_db_path)
        return None
