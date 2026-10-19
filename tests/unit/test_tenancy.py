"""Unit tests for Enterprise Tenancy, Commercialization, and Admin Authorization Gates."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.core.config import Settings
from sentinel.core.tenancy import EnterpriseCreateRequest, EnterpriseManager
from sentinel.llm.fake import FakeProvider
from sentinel.main import build_app
from tests.conftest import InMemoryResponseCache


@pytest.fixture
def fake_llm() -> FakeProvider:
    return FakeProvider(
        default_response="Deployments ensure zero-downtime rolling updates using maxSurge and maxUnavailable.",
        provider_name="fake-gemini",
        model="gemini-3.8-flash",
    )


@pytest.mark.asyncio
async def test_enterprise_manager_lifecycle():
    """Test core EnterpriseManager operations."""
    manager = EnterpriseManager()

    # Default seeded enterprises exist
    enterprises = await manager.list_enterprises()
    assert len(enterprises) >= 3

    # Onboard a new enterprise
    req = EnterpriseCreateRequest(
        name="Test FinTech AI",
        contact_email="dev@testfintech.ai",
        plan="growth",
        rate_limit_per_minute=500,
        notes="High frequency financial RAG testing",
    )
    ent = await manager.create_enterprise(req)
    assert ent.name == "Test FinTech AI"
    assert ent.status == "pending"
    assert ent.production_allowed is False
    assert ent.test_allowed is True
    assert ent.test_api_key.startswith("sk_test_")
    assert ent.live_api_key.startswith("sk_live_")

    # Validate test key -> should be authorized for test environment
    is_valid, matched_ent, env, is_authorized, err = manager.validate_key(ent.test_api_key)
    assert is_valid is True
    assert env == "test"
    assert is_authorized is True

    # Validate live key before approval -> should NOT be authorized
    is_valid, matched_ent, env, is_authorized, err = manager.validate_key(ent.live_api_key)
    assert is_valid is True
    assert env == "production"
    assert is_authorized is False
    assert "pending production authorization" in err

    # Admin authorizes the enterprise for production
    authorized_ent = await manager.authorize_enterprise(
        ent.id, admin_user="admin@sentinel.ai", rate_limit=1000
    )
    assert authorized_ent is not None
    assert authorized_ent.status == "authorized"
    assert authorized_ent.production_allowed is True
    assert authorized_ent.rate_limit_per_minute == 1000

    # Validate live key after approval -> should now be authorized!
    is_valid, _, env, is_authorized, _ = manager.validate_key(ent.live_api_key)
    assert is_valid is True
    assert env == "production"
    assert is_authorized is True

    # Admin revokes authorization
    revoked_ent = await manager.revoke_enterprise(ent.id, admin_user="security@sentinel.ai")
    assert revoked_ent is not None
    assert revoked_ent.status == "suspended"
    assert revoked_ent.production_allowed is False

    is_valid, _, _, is_authorized, err = manager.validate_key(ent.live_api_key)
    assert is_authorized is False

    # Key rotation
    old_test_key = ent.test_api_key
    updated_ent, new_test_key = await manager.rotate_key(ent.id, "test")
    assert new_test_key != old_test_key
    assert updated_ent.test_api_key == new_test_key
    assert new_test_key.startswith("sk_test_")


@pytest.mark.asyncio
async def test_admin_api_endpoints(fake_llm: FakeProvider, in_memory_cache: InMemoryResponseCache):
    """Test /v1/admin/enterprises endpoints."""
    settings = Settings(
        api_key_auth_enabled=True,
        cache_enabled=False,
        semantic_cache_enabled=False,
        guardrails_enabled=False,
        groundedness_gate_enabled=False,
    )

    app = build_app(settings=settings, override_primary_llm=fake_llm)
    app.state.cache = in_memory_cache
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "answer": "Deployments manage rolling updates.",
            "sources": [],
            "meta": {"tokens_in": 5, "tokens_out": 5, "latency_ms": 10.0},
        }
    )
    app.state.agent_graph = mock_graph

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. List enterprises
        res = await client.get("/v1/admin/enterprises")
        assert res.status_code == 200
        enterprises = res.json()
        assert len(enterprises) >= 3

        # 2. Create enterprise
        create_res = await client.post(
            "/v1/admin/enterprises",
            json={
                "name": "OmniCorp Health",
                "contact_email": "security@omnicorp.com",
                "plan": "enterprise",
                "rate_limit_per_minute": 1500,
            },
        )
        assert create_res.status_code == 201
        ent = create_res.json()
        ent_id = ent["id"]
        assert ent["status"] == "pending"

        # 3. Diagnostic Test Query in Sandbox environment -> should succeed (200 OK)
        diag_test = await client.post(
            "/v1/admin/test-query",
            json={
                "enterprise_id": ent_id,
                "environment": "test",
                "question": "How do Deployments work?",
            },
        )
        assert diag_test.status_code == 200
        data_test = diag_test.json()
        assert data_test["authorized"] is True
        assert data_test["environment"] == "test"

        # 4. Diagnostic Test Query in Production environment -> should be blocked by Authorization Gate!
        diag_prod_blocked = await client.post(
            "/v1/admin/test-query",
            json={
                "enterprise_id": ent_id,
                "environment": "production",
                "question": "How do Deployments work?",
            },
        )
        assert diag_prod_blocked.status_code == 200
        data_blocked = diag_prod_blocked.json()
        assert data_blocked["authorized"] is False
        assert data_blocked["status_code"] == 403
        assert "Authorization Denied" in data_blocked["error"]

        # 5. Direct live call to /v1/chat using live_api_key -> HTTP 403 Forbidden
        live_key = ent["live_api_key"]
        chat_blocked = await client.post(
            "/v1/chat",
            json={"question": "What is Kubernetes?"},
            headers={"Authorization": f"Bearer {live_key}"},
        )
        assert chat_blocked.status_code == 403
        assert "pending production authorization" in chat_blocked.json()["detail"]

        # 6. Admin authorizes the enterprise
        auth_res = await client.post(
            f"/v1/admin/enterprises/{ent_id}/authorize",
            json={"admin_user": "admin@sentinel.io", "rate_limit_per_minute": 2000},
        )
        assert auth_res.status_code == 200
        assert auth_res.json()["status"] == "authorized"
        assert auth_res.json()["production_allowed"] is True

        # 7. Diagnostic Test Query in Production environment -> now succeeds (200 OK)
        diag_prod_allowed = await client.post(
            "/v1/admin/test-query",
            json={
                "enterprise_id": ent_id,
                "environment": "production",
                "question": "How do Deployments work?",
            },
        )
        assert diag_prod_allowed.status_code == 200
        assert diag_prod_allowed.json()["authorized"] is True

        # 8. Rotate live key
        rotate_res = await client.post(
            f"/v1/admin/enterprises/{ent_id}/keys/rotate?environment=production"
        )
        assert rotate_res.status_code == 200
        new_key = rotate_res.json()["new_key"]
        assert new_key != live_key
