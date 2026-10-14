# SPDX-License-Identifier: Apache-2.0
"""Retired cluster admin routes must remain unavailable."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin.cluster_routes import cluster_router


@pytest.mark.parametrize("method, path", [
    ("get", "/config"), ("post", "/config"),
    ("get", "/status"), ("post", "/control"),
])
def test_cluster_admin_routes_are_retired(method, path):
    app = FastAPI()
    app.include_router(cluster_router)
    response = getattr(TestClient(app), method)("/admin/api/cluster" + path)
    assert response.status_code == 404
