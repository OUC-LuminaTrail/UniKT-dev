"""Format Python code blocks inside Markdown (MyST) files with ruff.

Targets ```` ```python ```` fenced blocks.

Usage:
    python scripts/fmt-code-blocks.py [path] [--check]
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE_OPEN = re.compile(r"^(\s*)```python\s*$")
FENCE_CLOSE = re.compile(r"^(\s*)```\s*$")
RUFF = "ruff"


def format_python(code: str) -> str:
    """Format a Python code block with ruff."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as tf:
        tf.write(code)
        tf.flush()
        tmp = tf.name
    try:
        subprocess.run(
            [RUFF, "format", "--quiet", tmp], capture_output=True, timeout=10
        )
        return Path(tmp).read_text().rstrip("\n")
    finally:
        Path(tmp).unlink(missing_ok=True)


def process_file(path: Path) -> bool:
    """Return True if the file was modified."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    i = 0
    modified = False
    while i < len(lines):
        if not FENCE_OPEN.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        out.append(lines[i])  # keep the ```python fence
        i += 1
        block: list[str] = []
        while i < len(lines) and not FENCE_CLOSE.match(lines[i]):
            block.append(lines[i])
            i += 1
        original = "\n".join(block)
        formatted = format_python(original)
        if formatted != original:
            modified = True
        out.extend(formatted.split("\n"))
        if i < len(lines):  # closing fence
            out.append(lines[i])
            i += 1
    new_text = "\n".join(out)
    if not new_text.endswith("\n"):
        new_text += "\n"
    if new_text != path.read_text(encoding="utf-8"):
        path.write_text(new_text, encoding="utf-8")
        return True
    return modified


def main() -> int:
    """Run the script, returning 0 on success and 1 on failure."""
    check_only = "--check" in sys.argv
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else Path("source")
    )
    files = sorted(root.rglob("*.md"))
    changed = sum(1 for f in files if process_file(f))
    if check_only and changed:
        print(f"{changed} file(s) would be formatted. Run without --check to fix.")
        sys.exit(1)
    print(f"Formatted: {changed} file(s)")
    return 0


if __name__ == "__main__":
    main()
