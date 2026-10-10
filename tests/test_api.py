from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_pipeline import FixtureClient

from loan_intake import api
from loan_intake.store import Store


def test_upload_review_and_queue(tmp_path, fixture_dirs, monkeypatch):
    folder = next(p for p in fixture_dirs if FixtureClient(p).expected["expected_decision"] == "needs_review")
    store = Store(tmp_path / "test.db")
    api.app.dependency_overrides[api.get_store] = lambda: store
    api.app.dependency_overrides[api.get_client] = lambda: FixtureClient(folder)
    client = TestClient(api.app)
    files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in sorted(folder.glob("*.pdf"))]

    created = client.post("/dossiers", files=files).json()
    assert created["decision"] == "needs_review"
    card = next(d for d in created["documents"] if d["type"] == "national_id")
    assert "*" in card["id_number"]

    queue = client.get("/review-queue").json()
    assert [q["id"] for q in queue] == [created["id"]]

    reviewed = client.post(f"/dossiers/{created['id']}/review",
                           json={"reviewer": "analyst-01", "action": "accept", "note": "checked with branch"}).json()
    assert reviewed["status"] == "ready"
    assert client.get("/review-queue").json() == []
    record = client.get(f"/dossiers/{created['id']}").json()
    assert record["reviews"][0]["reviewer"] == "analyst-01"
    api.app.dependency_overrides.clear()


def test_unsupported_file_type_is_rejected(tmp_path):
    api.app.dependency_overrides[api.get_store] = lambda: Store(tmp_path / "test.db")
    api.app.dependency_overrides[api.get_client] = lambda: None
    response = TestClient(api.app).post("/dossiers", files=[("files", ("note.docx", b"x", "application/octet-stream"))])
    assert response.status_code == 415
    api.app.dependency_overrides.clear()


@pytest.mark.parametrize("name", ["scan.png", "form.pdf"])
def test_damaged_file_is_rejected(tmp_path, name):
    api.app.dependency_overrides[api.get_store] = lambda: Store(tmp_path / "test.db")
    api.app.dependency_overrides[api.get_client] = lambda: None
    response = TestClient(api.app).post("/dossiers", files=[("files", (name, b"not really a file", "image/png"))])
    assert response.status_code == 422
    api.app.dependency_overrides.clear()
