"""Tests for the schema reflection helper's docstring-help extraction.

Covers the Google-style fallback parser used when ``docstring_parser`` is
absent (conda / custom-interpreter environments; the dhg pixi feature
declares the package but the fallback stays as the safety net), its parity
with the real parser on every reflected config class, and the degraded-mode
dispatch.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import pytest

import web.backend.services._schema_helper as helper


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

    def test_multiparagraph_entry_keeps_later_paragraphs(self):
        # Blank lines inside a description are dropped when a deeper-indented
        # continuation follows, matching docstring_parser's single-newline join.
        doc = "S.\n\nArgs:\n    a: first para.\n\n        second para.\n"
        assert helper._fallback_docstring_helps(doc) == {
            "a": "first para.\nsecond para."
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


class TestParserParity:
    def test_fallback_matches_docstring_parser_on_all_config_classes(
        self, registry_snapshot
    ):
        """Every reflected config class (framework + all registered models)
        must expose identical per-field helps under both parsers."""
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        import model  # noqa: F401  triggers @register_model_config discovery
        from utils import config as cfg
        from utils.core import MODEL_CONFIGS, get_supported_models

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
        ] + [MODEL_CONFIGS.get(name) for name in get_supported_models()]
        classes = [c for c in classes if c is not None]

        checked = 0
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
            checked += 1
        assert checked > 70  # 6 framework + the full model registry
