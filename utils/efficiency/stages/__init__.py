"""Efficiency stages package.

Importing this package triggers static discovery: ``@register_efficiency_stage``
calls under this directory are scanned (without importing stage modules) and
indexed into :data:`utils.core.EFFICIENCY_STAGES`. Stage classes are imported
lazily on ``EFFICIENCY_STAGES.get(name)``. Add a stage by dropping a file here
that decorates a class — no manual registration.
"""

from pathlib import Path

from utils.core import discover_registrations

discover_registrations(Path(__file__).parent, "utils.efficiency.stages")
