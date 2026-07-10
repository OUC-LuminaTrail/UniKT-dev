"""Format Python code blocks inside reStructuredText files with ruff.

Targets ``.. code:: python`` and ``.. code-block:: python`` directives (both forms
appear in the markdown -> rST conversion output).

Usage:
    python scripts/fmt-code-blocks.py [path] [--check]
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_DIRECTIVE = re.compile(r"^\s*\.\. code(?:-block)?:: python\b")
RUFF = "ruff"  # resolved from PATH; install via your approved toolchain


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
    """Returns True if the file was modified."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    i = 0
    modified = False
    while i < len(lines):
        if not CODE_DIRECTIVE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        out.append(lines[i])  # keep the directive line
        i += 1
        # preserve blank lines between the directive and its indented body
        while i < len(lines) and lines[i].strip() == "":
            out.append(lines[i])
            i += 1
        if i >= len(lines) or not lines[i].startswith(" "):
            continue  # directive has no indented body

        base = len(lines[i]) - len(lines[i].lstrip(" "))
        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                block.append("")
                i += 1
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent < base:
                break
            block.append(line[base:])
            i += 1

        while block and block[-1] == "":
            block.pop()
        original = "\n".join(block)
        formatted = format_python(original)
        if formatted != original:
            modified = True
        for fl in formatted.split("\n"):
            out.append((" " * base + fl) if fl else "")
        # docutils requires a blank line after an explicit-markup block
        if i >= len(lines) or lines[i].strip() != "":
            out.append("")

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
    files = sorted(root.rglob("*.rst"))
    changed = sum(1 for f in files if process_file(f))
    if check_only and changed:
        print(f"{changed} file(s) would be formatted. Run without --check to fix.")
        sys.exit(1)
    print(f"Formatted: {changed} file(s)")
    return 0


if __name__ == "__main__":
    main()
