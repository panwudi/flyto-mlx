# SPDX-License-Identifier: Apache-2.0
"""Render and behavior contracts for the offline Flyto MLX admin UI."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ADMIN = Path(__file__).resolve().parents[1] / "omlx" / "admin"


@pytest.mark.parametrize("locale", sorted(p.stem for p in (ADMIN / "i18n").glob("*.json")))
def test_all_pages_render_with_local_branding_and_assets(locale):
    strings = json.loads((ADMIN / "i18n" / f"{locale}.json").read_text())
    env = Environment(loader=FileSystemLoader(ADMIN / "templates"), autoescape=True)
    context = dict(
        current_lang=locale,
        t=lambda key: strings[key],
        locale_json=json.dumps(strings),
        static=lambda path: "/admin/static/" + path,
        version="test",
        api_key_configured=True,
        api_key="fixture-key",
    )
    for page in ("dashboard.html", "chat.html", "login.html"):
        html = env.get_template(page).render(**context)
        assert "Flyto MLX" in html
        assert "omlx.ai" not in html
        assert "jundot/omlx" not in html
        assert 'src="https://' not in html
        assert 'href="https://fonts.' not in html
        assert "/admin/static/css/flyto.css" in html
    dashboard = env.get_template("dashboard.html").render(**context)
    assert 'id="admin-navigation"' in dashboard
    assert dashboard.index('class="flyto-runtime"') < dashboard.index('class="flyto-disclosure"')
    assert 'role="status" aria-live="polite"' in dashboard


def test_dashboard_behavior_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the dashboard JavaScript contracts")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("test_admin_ui.js")), str(ADMIN)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_removed_remote_preset_route_and_local_results(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.admin import benchmark, routes

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.require_admin] = lambda: True
    client = TestClient(app)
    assert client.post("/admin/api/presets/refresh").status_code == 404
    run = benchmark.BenchmarkRun(
        bench_id="local-result",
        request=benchmark.BenchmarkRequest(model_id="fixture", prompt_lengths=[1024]),
        status="completed",
        results=[{"test_type": "single", "pp": 1024, "gen_tps": 42}],
    )
    monkeypatch.setattr(benchmark, "get_run", lambda _: run)
    response = client.get("/admin/api/bench/local-result/results")
    assert response.status_code == 200
    assert response.json()["results"] == run.results
    assert "upload_state" not in response.json()


def test_device_info_contains_hardware_without_owner_identity(monkeypatch):
    import asyncio

    from omlx.admin import routes
    from omlx.utils import hardware

    monkeypatch.setattr(hardware, "get_chip_name", lambda: "Apple M5 Max")
    monkeypatch.setattr(hardware, "get_gpu_core_count", lambda: 40)
    monkeypatch.setattr(hardware, "get_total_memory_gb", lambda: 128)

    def forbidden_identity():
        raise AssertionError("Local benchmark display must not read the device UUID")

    monkeypatch.setattr(hardware, "get_io_platform_uuid", forbidden_identity)
    result = asyncio.run(routes.get_device_info(is_admin=True))
    assert result["memory_gb"] == 128
    assert result["gpu_cores"] == 40
    assert "owner_hash" not in result
