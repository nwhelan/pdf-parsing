from __future__ import annotations

import pytest

from pdfplay.samples import make_bank_statement
from pdfplay.workspace import Workspace


@pytest.fixture(scope="session")
def borderless(tmp_path_factory):
    """A borderless synthetic statement plus its ledger."""
    out = tmp_path_factory.mktemp("samples") / "borderless.pdf"
    return make_bank_statement(out, variant="borderless", seed=11, n_transactions=40)


@pytest.fixture(scope="session")
def ruled(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples") / "ruled.pdf"
    return make_bank_statement(out, variant="ruled", seed=11, n_transactions=40)


@pytest.fixture(scope="session")
def scanned(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples") / "scanned.pdf"
    return make_bank_statement(out, variant="scanned", seed=11, n_transactions=40)


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "ws")
