"""Tests for the schema reflection helper's docstring-help extraction.

Covers the Google-style fallback parser used when ``docstring_parser`` is
absent (conda / custom-interpreter environments; the dhg pixi feature
declares the package but the fallback stays as the safety net), its parity
with the real parser on every reflected config class, the degraded-mode
dispatch and signal, and the module's import side-effect hygiene.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, fields
from typing import ClassVar

import pytest

import web.backend.services._schema_helper as helper

_REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


def _cls_with_doc(doc):
    @dataclass
    class WithDoc:
        x: int = 0

    WithDoc.__doc__ = doc
    return WithDoc


class TestFallbackParser:
    def test_single_line_entries(self):
        doc = "Summary.\n\nArgs:\n    a: help a.\n    b: help b.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a.", "b": "help b."}

    def test_multiline_description_joined_with_newlines(self):
        doc = "S.\n\nArgs:\n    a: first line\n        second line\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "first line\nsecond line"}

    def test_typed_entry_form(self):
        doc = "S.\n\nArgs:\n    a (int): typed help.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "typed help."}

    def test_typed_entry_nested_parens(self):
        # One nesting level inside the type annotation must not break the
        # entry match (docstring_parser parses 'tuple(int, int)' fine).
        doc = "S.\n\nArgs:\n    bins (tuple(int, int)): bin edges.\n"
        assert helper._fallback_docstring_helps(doc) == {"bins": "bin edges."}

    def test_section_aliases(self):
        # docstring_parser accepts these Google-style aliases; the fallback
        # must too or the whole section silently yields {}.
        for alias in ("Parameters", "Arguments", "Params", "Attributes"):
            doc = f"S.\n\n{alias}:\n    a: help a.\n"
            assert helper._fallback_docstring_helps(doc) == {"a": "help a."}, alias

    def test_trailing_note_section_not_an_entry(self):
        # A "Note:" paragraph indented like entries (HGKTConfig style) must
        # not register a bogus "Note" param.
        doc = (
            "S.\n\nArgs:\n    a: help a.\n\n"
            "    Note: no scheduler here,\n    unlike elsewhere.\n"
        )
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}

    def test_section_resumes_after_mid_note(self):
        # Entries documented after a Note:-style paragraph keep their helps
        # (docstring_parser also keeps them).
        doc = "S.\n\nArgs:\n    a: aa\n\n    Note: mid note\n\n    b: bb"
        assert helper._fallback_docstring_helps(doc) == {"a": "aa", "b": "bb"}

    def test_deeper_entries_survive_shallower_note(self):
        # Entries indented deeper than the Note: line must not be swallowed
        # as note text — a blank line ends the note paragraph.
        doc = "S.\n\nArgs:\n        a: aa\n\n    Note: see the paper.\n\n        b: bb"
        helps = helper._fallback_docstring_helps(doc)
        assert helps.get("b") == "bb"

    def test_multiparagraph_entry_blanking_rules(self):
        # docstring_parser joins the entry line (short) and the deeper lines
        # (long) with a single newline; the leading blank is dropped and
        # inner blanks are kept.
        one_blank = "S.\n\nArgs:\n    a: p1.\n\n        p2.\n"
        assert helper._fallback_docstring_helps(one_blank) == {"a": "p1.\np2."}
        two_blanks = "S.\n\nArgs:\n    a: p1.\n\n        p2.\n\n        p3.\n"
        assert helper._fallback_docstring_helps(two_blanks) == {"a": "p1.\np2.\n\np3."}
        no_leading_blank = "S.\n\nArgs:\n    a: p1.\n        p2.\n\n        p3.\n"
        assert helper._fallback_docstring_helps(no_leading_blank) == {
            "a": "p1.\np2.\n\np3."
        }

    def test_deeper_entry_like_line_after_blank_joins_description(self):
        # Not a new entry: it would dict-overwrite and attach the WRONG help.
        doc = "S.\n\nArgs:\n    a: real help.\n\n        clarification: more\n"
        assert helper._fallback_docstring_helps(doc) == {
            "a": "real help.\nclarification: more"
        }

    def test_empty_and_missing_docstring(self):
        assert helper._fallback_docstring_helps("") == {}
        assert helper._fallback_docstring_helps("No sections at all.") == {}

    def test_non_entry_lines_in_section_skipped(self):
        doc = "S.\n\nArgs:\n    free text line\n    a: help a.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}


class TestDegradedMode:
    def test_missing_parser_falls_back(self, monkeypatch):
        monkeypatch.setattr(helper, "_parse_docstring", None)
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "help x."}

    def test_installed_parser_is_consulted(self, monkeypatch):
        # A distinctive stub value proves dispatch reaches the installed
        # parser — the fallback alone would return the docstring's own text.
        class _StubParam:
            arg_name = "x"
            description = "stub value"

        class _StubDoc:
            params: ClassVar = [_StubParam()]

        monkeypatch.setattr(helper, "_parse_docstring", lambda doc: _StubDoc())
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "stub value"}

    def test_degraded_marker_emitted_on_stdout(self, monkeypatch, capsys):
        # The marker rides the stdout envelope protocol (stderr is discarded
        # by schema_extractor on success paths).
        monkeypatch.setattr(helper, "_parse_docstring", None)
        helper._emit_degraded_marker()
        out = capsys.readouterr().out.strip()
        assert json.loads(out) == {"type": "meta", "degraded": True}

    def test_no_marker_when_parser_installed(self, capsys):
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        helper._emit_degraded_marker()
        assert capsys.readouterr().out == ""


class TestImportHygiene:
    def test_import_does_not_prepend_cwd_to_sys_path(self):
        # Module-scope sys.path.insert would leak into any importing process
        # (pytest); the insert must live in the __main__ block only.
        code = (
            "import sys; "
            f"sys.path.insert(0, {_REPO_ROOT!r}); "
            "import web.backend.services._schema_helper; "
            "print(sys.path[0])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() != "."


class TestParserParity:
    def test_fallback_matches_docstring_parser_on_all_config_classes(
        self, registry_snapshot
    ):
        """Every reflected config class (framework + all registered models)
        must expose identical per-field helps under both parsers."""
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        try:
            import model  # noqa: F401  triggers @register_model_config discovery
            from utils.core import MODEL_CONFIGS, get_supported_models

            model_names = get_supported_models()
            model_classes = [MODEL_CONFIGS.get(name) for name in model_names]
        except ImportError as e:
            # Trainer modules import torch, which torch-less-but-parser-having
            # environments (web, docs) lack; parity over models needs it.
            pytest.skip(f"training stack unavailable: {e}")
        from utils import config as cfg

        classes = [
            getattr(cfg, name)
            for name in (
                "CompileConfig",
                "DownloadConfig",
                "EarlyStoppingConfig",
                "GeneralConfig",
                "ProcessConfig",
                "RunDataConfig",
            )
        ] + model_classes
        assert len(classes) == 6 + len(model_names)

        for cls in classes:
            reference = helper._parse_docstring_helps(cls)
            fallback = helper._fallback_docstring_helps(cls.__doc__ or "")
            for f in fields(cls):
                assert fallback.get(f.name) == reference.get(f.name), (
                    cls.__name__,
                    f.name,
                )
            # Ghost keys beyond real fields must still come from the reference
            # parse (the fallback never invents entries docstring_parser lacks).
            assert set(fallback) <= set(reference) | {f.name for f in fields(cls)}, (
                cls.__name__
            )
