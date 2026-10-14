# SPDX-License-Identifier: Apache-2.0
"""Compatibility router after retirement of cluster admin management.

The API server still includes this router. It intentionally exposes no routes;
the standalone cluster router and its inference behavior remain independent.
"""

from fastapi import APIRouter

cluster_router = APIRouter(prefix="/admin/api/cluster")
