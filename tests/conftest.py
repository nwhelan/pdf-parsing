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


@pytest.fixture
def client(workspace: Workspace, borderless):
    """A TestClient over a workspace holding one scored document."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from pdfplay.server.app import create_app

    workspace.add_document(borderless.path, doc_class="bank_statement")
    workspace.set_ground_truth(
        workspace.list_documents()[0].doc_id, {"transactions": borderless.ledger["transactions"]}
    )
    return fastapi_testclient.TestClient(create_app(workspace))
