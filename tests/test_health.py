"""`/health` reports per-dependency status and never lies about being up."""

from __future__ import annotations

import asyncio
import time

import pytest

from nsu_rag_llm.api.deps import measure
from tests.conftest import fixed_probe


def test_all_dependencies_up(client, set_probes) -> None:
    set_probes(fixed_probe(detail="2 collection(s)"), fixed_probe(detail="model served"))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant"]["status"] == "ok"
    assert body["llm"]["status"] == "ok"
    assert body["qdrant"]["detail"] == "2 collection(s)"


def test_qdrant_down_is_reported_as_unavailable(client, set_probes) -> None:
    set_probes(fixed_probe(status="error", detail="ConnectError: refused"), fixed_probe())

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["qdrant"]["status"] == "error"
    # The healthy dependency must still be reported as healthy, so that the
    # response says which half of the system to go and look at.
    assert body["llm"]["status"] == "ok"


def test_llm_down_is_reported_as_unavailable(client, set_probes) -> None:
    set_probes(fixed_probe(), fixed_probe(status="error", detail="model not served"))

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["llm"]["status"] == "error"
    assert body["qdrant"]["status"] == "ok"


def test_probes_run_concurrently(client, set_probes) -> None:
    """Two slow probes must cost about one probe's worth of wall clock."""
    delay = 0.2
    set_probes(fixed_probe(delay=delay), fixed_probe(delay=delay))

    started = time.perf_counter()
    response = client.get("/health")
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < delay * 1.75, f"probes look sequential: {elapsed:.3f}s"


async def test_measure_reports_a_raised_error_instead_of_propagating() -> None:
    async def boom() -> str:
        raise ConnectionRefusedError("nothing listening on 6333")

    result = await measure(boom, timeout=1.0)

    assert result.status == "error"
    assert "ConnectionRefusedError" in (result.detail or "")
    assert "nothing listening" in (result.detail or "")


async def test_measure_gives_up_on_a_hung_dependency() -> None:
    async def hangs() -> str:
        await asyncio.sleep(10)
        return "never"

    started = time.perf_counter()
    result = await measure(hangs, timeout=0.1)
    elapsed = time.perf_counter() - started

    assert result.status == "error"
    assert "timed out" in (result.detail or "")
    assert elapsed < 1.0


async def test_measure_records_latency_for_a_healthy_check() -> None:
    async def quick() -> str:
        await asyncio.sleep(0.05)
        return "all good"

    result = await measure(quick, timeout=1.0)

    assert result.status == "ok"
    assert result.detail == "all good"
    assert result.latency_ms >= 50


@pytest.mark.parametrize("path", ["/health"])
def test_health_body_shape_is_stable(client, set_probes, path: str) -> None:
    set_probes(fixed_probe(), fixed_probe())

    body = client.get(path).json()

    assert set(body) == {"status", "qdrant", "llm"}
    assert set(body["qdrant"]) == {"status", "latency_ms", "detail"}
