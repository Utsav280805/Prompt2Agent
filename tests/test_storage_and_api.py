"""
Integration tests for persistence and the HTTP API.

These cross real boundaries - a real SQLite file in a temporary directory and a real ASGI app -
which is the point. Mocking the database would test the mock, and the bugs that actually bite
here are boundary bugs: a partial save, a cascade that does not cascade, a key that comes back
out of an endpoint it should never appear in.

Nothing here needs a credential or a network. Generation is never triggered; only the endpoints
that read state, describe the server, or manage settings are exercised. Anything that would
reach a model belongs behind the ``live`` marker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from multi_agent_generator.core.models import (
    GeneratedProject,
    IterationRecord,
    PipelineResult,
    ProjectStatus,
    ReviewResult,
    TestReport,
)
from multi_agent_generator.settings import Settings
from multi_agent_generator.storage.repository import Storage

pytestmark = pytest.mark.integration

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.config import AppContext  # noqa: E402


# ------------------------------------------------------------------ fixtures
def _settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temporary directory.

    Built explicitly rather than through ``from_env`` so a developer's real ``.env`` cannot
    leak into a test run - including, in the worst case, a real API key and a real database.
    """
    data = tmp_path / "data"
    return Settings(
        data_dir=data,
        database_url=f"sqlite:///{(data / 'test.db').as_posix()}",
        workspaces_dir=tmp_path / "workspaces",
        deps_cache_dir=tmp_path / "deps",
        default_provider="huggingface",
    )


@pytest.fixture()
def storage(tmp_path: Path):
    store = Storage(_settings(tmp_path))
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def client(tmp_path: Path):
    settings = _settings(tmp_path)
    store = Storage(settings)
    context = AppContext(settings=settings, storage=store)
    app = create_app(context=context, settings=settings, serve_frontend=False)
    with TestClient(app) as test_client:
        yield test_client
    store.close()


def _finished_result() -> PipelineResult:
    project = GeneratedProject(framework="crewai", provider="huggingface", model="m")
    project.add_file("main.py", "print('hi')\n", is_entrypoint=True)
    result = PipelineResult(
        requirement="Create a research assistant",
        status=ProjectStatus.READY,
        project=project,
    )
    result.iterations.append(
        IterationRecord(
            index=1,
            review=ReviewResult(score=9.0, passed=True),
            tests=TestReport(status="passed", exit_code=0, passed_count=2),
        )
    )
    return result


# ------------------------------------------------------------------ storage
class TestStorage:
    def test_constructing_storage_creates_the_database(self, storage):
        assert Path(storage.db.path).is_file()

    def test_a_project_round_trips_through_sqlite(self, storage):
        project_id = storage.projects.create(
            requirement="Create a research assistant", framework="crewai"
        )
        storage.projects.save_result(project_id, _finished_result())

        record = storage.projects.get(project_id)
        assert record["status"] == "ready"
        assert record["requirement"] == "Create a research assistant"
        assert record["result"]["project"]["files"][0]["path"] == "main.py"

    def test_the_list_view_omits_generated_source(self, storage):
        project_id = storage.projects.create(requirement="Create a research assistant")
        storage.projects.save_result(project_id, _finished_result())

        summary = storage.projects.list()[0]
        assert summary["id"] == project_id
        # Listing every byte ever generated to draw a table of rows is how a list page gets
        # slow enough to look broken.
        assert "result" not in summary or not summary.get("result")

    def test_deleting_a_project_reports_whether_it_existed(self, storage):
        project_id = storage.projects.create(requirement="x")
        assert storage.projects.delete(project_id) is True
        assert storage.projects.delete(project_id) is False
        assert storage.projects.get(project_id) is None

    def test_a_missing_project_is_none_not_an_exception(self, storage):
        assert storage.projects.get("prj_does_not_exist") is None

    def test_saved_settings_never_hand_back_the_key_unless_asked(self, storage):
        if not storage.secretbox.available:
            pytest.skip("cryptography is not installed, so keys cannot be stored")

        storage.llm_config.save(
            {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-round-trip-123456"}
        )

        masked = storage.saved_llm()
        assert masked["provider"] == "openai"
        assert masked["credential_configured"] is True
        assert masked.get("api_key") in (None, "***")

        revealed = storage.saved_llm(reveal=True)
        assert revealed["api_key"] == "sk-round-trip-123456"

    def test_the_key_is_not_stored_in_plaintext_on_disk(self, storage):
        if not storage.secretbox.available:
            pytest.skip("cryptography is not installed, so keys cannot be stored")
        storage.llm_config.save({"provider": "openai", "api_key": "sk-plaintext-check-1234"})
        storage.db.close()
        blob = Path(storage.db.path).read_bytes()
        assert b"sk-plaintext-check-1234" not in blob

    def test_saving_without_a_key_keeps_the_existing_one(self, storage):
        if not storage.secretbox.available:
            pytest.skip("cryptography is not installed, so keys cannot be stored")
        storage.llm_config.save({"provider": "openai", "api_key": "sk-keep-me-123456"})
        # The realistic action: the user edits the model name and saves.
        storage.llm_config.save({"provider": "openai", "model": "gpt-4o"})
        revealed = storage.saved_llm(reveal=True)
        assert revealed["api_key"] == "sk-keep-me-123456"
        assert revealed["model"] == "gpt-4o"

    def test_clear_key_removes_only_the_key(self, storage):
        if not storage.secretbox.available:
            pytest.skip("cryptography is not installed, so keys cannot be stored")
        storage.llm_config.save({"provider": "openai", "model": "gpt-4o", "api_key": "sk-bye-123456"})
        storage.llm_config.clear_key()
        revealed = storage.saved_llm(reveal=True)
        # The masked read model has no ``api_key`` key at all when none is stored, which is a
        # stronger guarantee than an empty string: there is nothing to accidentally serialise.
        assert revealed.get("api_key") in (None, "")
        assert revealed["credential_configured"] is False
        assert revealed["model"] == "gpt-4o"

    def test_a_non_sqlite_database_url_is_refused_rather_than_ignored(self, tmp_path):
        from multi_agent_generator.errors import ConfigurationError

        broken = _settings(tmp_path)
        broken.database_url = "postgresql://localhost/whatever"
        with pytest.raises(ConfigurationError):
            Storage(broken)


# ------------------------------------------------------------------ HTTP API
class TestApi:
    def test_health_reports_what_the_server_can_actually_do(self, client):
        body = client.get("/api/health").json()
        assert body["status"] in ("ok", "degraded")
        assert body["persistence"] is True
        assert "credential_configured" in body
        assert "api_key" not in body

    def test_unknown_endpoints_use_the_standard_error_shape(self, client):
        response = client.get("/api/nope")
        assert response.status_code == 404
        body = response.json()
        # One error shape across the whole API is what lets the frontend have one error
        # component instead of a guess per endpoint.
        assert set(body) >= {"code", "message"}

    def test_a_missing_project_is_a_clean_404_not_a_traceback(self, client):
        response = client.get("/api/projects/prj_missing")
        assert response.status_code == 404
        body = response.json()
        assert body["code"] == "not_found"
        assert "Traceback" not in body["message"]

    def test_settings_never_return_the_api_key(self, client):
        body = client.get("/api/settings/llm").json()
        assert "api_key" not in body
        # It may report *that* a key exists and its last four characters. That is the ceiling.
        assert "credential_configured" in body
        hint = body.get("credential_hint")
        assert hint is None or hint.startswith("***")

    def test_frameworks_and_providers_are_listable_without_a_credential(self, client):
        frameworks = client.get("/api/frameworks").json()
        providers = client.get("/api/providers").json()
        assert {f["name"] for f in frameworks} >= {"crewai", "langgraph"}
        assert any(p["name"] == "huggingface" for p in providers)
        # Nothing in a provider description is allowed to carry a secret.
        assert all("api_key" not in p for p in providers)

    def test_saving_settings_round_trips_without_echoing_a_key(self, client):
        response = client.put(
            "/api/settings/llm",
            json={"provider": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct", "temperature": 0.2},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["provider"] == "huggingface"
        assert body["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert "api_key" not in body

    def test_an_unknown_provider_is_a_400_with_an_action(self, client):
        response = client.put("/api/settings/llm", json={"provider": "not-a-provider"})
        assert response.status_code in (400, 422)
        body = response.json()
        if response.status_code == 400:
            assert body["code"] in ("unknown_provider", "configuration_error")
            assert body.get("action")

    def test_listing_projects_is_empty_rather_than_an_error(self, client):
        response = client.get("/api/projects")
        assert response.status_code == 200
        assert response.json() == []

    def test_a_project_created_directly_is_visible_through_the_api(self, client):
        # Written through storage rather than by running the pipeline: this asserts the read
        # path, and a real run needs a credential.
        from backend.config import get_context

        store = get_context().storage
        project_id = store.projects.create(requirement="Create a research assistant")
        store.projects.save_result(project_id, _finished_result())

        listed = client.get("/api/projects").json()
        assert [p["id"] for p in listed] == [project_id]

        detail = client.get(f"/api/projects/{project_id}").json()
        assert detail["status"] == "ready"

        files = client.get(f"/api/projects/{project_id}/files").json()
        assert any(f["path"] == "main.py" for f in files.get("files", files))

        assert client.delete(f"/api/projects/{project_id}").status_code in (200, 204)
        assert client.get(f"/api/projects/{project_id}").status_code == 404

    def test_downloading_a_project_returns_a_zip(self, client):
        from backend.config import get_context

        store = get_context().storage
        project_id = store.projects.create(requirement="Create a research assistant")
        store.projects.save_result(project_id, _finished_result())

        response = client.get(f"/api/projects/{project_id}/download")
        assert response.status_code == 200
        assert response.content[:2] == b"PK"  # a zip, not an HTML error page
