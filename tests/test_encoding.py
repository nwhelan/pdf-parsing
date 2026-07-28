"""Text I/O must not depend on the platform's default encoding.

Windows opens text files in the ANSI code page, so a parser that returns an em
dash, a curly quote or any non-Latin-1 character — which is to say every OCR
model on a real document — used to fail with "'charmap' codec can't encode
character" *after* the API call had already been paid for.

Linux can't reproduce cp1252, but it can reproduce the bug class: under
``LC_ALL=C`` the preferred encoding is ASCII, so an unqualified ``write_text``
raises exactly the same way. These tests run the real code in a subprocess with
that locale forced, which is the only way to prove the encoding is explicit
rather than merely convenient.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Characters no single-byte code page can hold, of the kind OCR output is full
# of: an em dash, curly quotes, an approximation sign, a CJK glyph.
AWKWARD = "— “quoted” ≈ 漢字 · €100"


def run_in_ascii_locale(code: str, tmp_path) -> subprocess.CompletedProcess:
    """Run a snippet with the ASCII locale that surfaces the bug.

    The snippet goes to a file rather than ``-c``: Python decodes source files
    as UTF-8 regardless of locale, but decodes ``-c`` using the locale, which
    would fail here for reasons that have nothing to do with what's under test.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "snippet.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",  # don't let UTF-8 mode paper over it
        "PYTHONCOERCECLOCALE": "0",  # nor PEP 538 coercion
        "PDFPLAY_WORKSPACE": str(tmp_path / "ws"),
    }
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=os.getcwd()
    )


@pytest.fixture(scope="module")
def ascii_locale_is_reachable(tmp_path_factory):
    """Skip if this interpreter refuses to give us a single-byte default."""
    probe = run_in_ascii_locale(
        "import locale; print(locale.getpreferredencoding(False))", tmp_path_factory.mktemp("probe")
    )
    encoding = probe.stdout.strip().lower()
    if "utf" in encoding:
        pytest.skip(f"cannot force a non-UTF-8 default encoding here (got {encoding!r})")
    return encoding


def test_a_result_full_of_awkward_characters_round_trips(tmp_path, ascii_locale_is_reachable):
    """The regression: saving an OCR result used to raise on the write."""
    done = run_in_ascii_locale(
        f"""
        from pdfplay.models import PageResult, ParseResult
        from pdfplay.workspace import Workspace

        text = {AWKWARD!r}
        ws = Workspace()
        result = ParseResult(
            parser_id="mistral-ocr-3",
            doc_id="deadbeef",
            markdown=text,
            pages=[PageResult(page_number=1, width=612, height=792, text=text)],
        )
        ws.save_result(result, "key1")
        loaded = ws.load_result("deadbeef", "key1")
        assert loaded is not None, "the result did not load back"
        assert loaded.markdown == text, "the text changed on the way through disk"
        print("ROUND TRIPPED")
        """,
        tmp_path,
    )
    assert "ROUND TRIPPED" in done.stdout, done.stderr
    assert "charmap" not in done.stderr and "UnicodeEncodeError" not in done.stderr


def test_document_metadata_and_ground_truth_survive_too(tmp_path, ascii_locale_is_reachable):
    done = run_in_ascii_locale(
        f"""
        from pdfplay.workspace import Workspace

        ws = Workspace()
        (ws.docs_dir / "doc1").mkdir(parents=True, exist_ok=True)
        ws.set_ground_truth("doc1", {{"note": {AWKWARD!r}}})
        assert ws.get_ground_truth("doc1")["note"] == {AWKWARD!r}
        print("GROUND TRUTH OK")
        """,
        tmp_path,
    )
    assert "GROUND TRUTH OK" in done.stdout, done.stderr


def test_the_cli_prints_its_tick_without_crashing(tmp_path, ascii_locale_is_reachable):
    """`pdfplay list` prints U+2713, which an ANSI console cannot encode."""
    done = run_in_ascii_locale(
        """
        import sys
        from pdfplay.cli import main
        sys.exit(main(["list"]))
        """,
        tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "UnicodeEncodeError" not in done.stderr
    assert "pymupdf" in done.stdout
