"""Tests for the schema reflection helper's docstring-help extraction.

Covers the Google-style fallback parser used when ``docstring_parser`` is
absent (dhg-gpu / dhg-cpu / custom interpreter environments), its parity
with the real parser when installed, and the degraded-mode dispatch.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

_HELPER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "web"
    / "backend"
    / "services"
    / "_schema_helper.py"
)


@pytest.fixture(scope="module")
def helper():
    """Load _schema_helper.py by path (it lives outside any importable package)."""
    spec = importlib.util.spec_from_file_location(
        "_schema_helper_under_test", _HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cls_with_doc(doc):
    @dataclass
    class WithDoc:
        x: int = 0

    WithDoc.__doc__ = doc
    return WithDoc


class TestFallbackParser:
    def test_single_line_entries(self, helper):
        doc = "Summary.\n\nArgs:\n    a: help a.\n    b: help b.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a.", "b": "help b."}

    def test_multiline_description_joined_with_newlines(self, helper):
        doc = "S.\n\nArgs:\n    a: first line\n        second line\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "first line\nsecond line"}

    def test_typed_entry_form(self, helper):
        doc = "S.\n\nArgs:\n    a (int): typed help.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "typed help."}

    def test_attributes_section_parsed(self, helper):
        # docstring_parser merges Attributes: entries into .params; the
        # fallback must too.
        doc = "S.\n\nArgs:\n    a: help a.\n\nAttributes:\n    b: help b.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a.", "b": "help b."}

    def test_trailing_note_section_not_an_entry(self, helper):
        # A "Note:" paragraph indented like entries (HGKTConfig style) must
        # terminate the Args block, not register a bogus "Note" param.
        doc = (
            "S.\n\nArgs:\n    a: help a.\n\n"
            "    Note: no scheduler here,\n    unlike elsewhere.\n"
        )
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}

    def test_empty_and_missing_docstring(self, helper):
        assert helper._fallback_docstring_helps("") == {}
        assert helper._fallback_docstring_helps("No sections at all.") == {}

    def test_non_entry_lines_in_section_skipped(self, helper):
        doc = "S.\n\nArgs:\n    free text line\n    a: help a.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}


class TestDegradedMode:
    def test_missing_parser_falls_back(self, helper, monkeypatch):
        monkeypatch.setattr(helper, "_parse_docstring", None)
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "help x."}

    def test_installed_parser_used_when_available(self, helper):
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "help x."}


class TestParserParity:
    def test_fallback_matches_docstring_parser_on_framework_configs(self, helper):
        """Every framework config's helps must be identical under both parsers."""
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        from utils.config import (
            CompileConfig,
            DownloadConfig,
            EarlyStoppingConfig,
            GeneralConfig,
            ProcessConfig,
            RunDataConfig,
        )

        for cls in (
            CompileConfig,
            DownloadConfig,
            EarlyStoppingConfig,
            GeneralConfig,
            ProcessConfig,
            RunDataConfig,
        ):
            reference = helper._parse_docstring_helps(cls)
            fallback = helper._fallback_docstring_helps(cls.__doc__ or "")
            assert fallback == reference, cls.__name__
