"""Enterprise Tenancy & Commercialization Management.

Supports multi-tenant enterprise onboarding, Test (Sandbox) vs Production environments,
API key generation (sk_test_ and sk_live_), and admin authorization gates.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class Enterprise(BaseModel):
    """Enterprise client profile."""

    id: str = Field(description="Unique enterprise identifier")
    name: str = Field(description="Company or organization name")
    contact_email: str = Field(description="Technical or admin contact email")
    plan: str = Field(default="growth", description="Subscription tier: trial, growth, enterprise")
    status: str = Field(default="pending", description="Status: pending, authorized, or suspended")
    test_api_key: str = Field(description="Sandbox / Test environment API key (sk_test_...)")
    live_api_key: str = Field(description="Production environment API key (sk_live_...)")
    test_allowed: bool = Field(
        default=True, description="Whether test / sandbox queries are permitted"
    )
    production_allowed: bool = Field(
        default=False, description="Whether live production queries are permitted"
    )
    test_requests_count: int = Field(default=0, description="Total sandbox requests executed")
    live_requests_count: int = Field(default=0, description="Total production requests executed")
    rate_limit_per_minute: int = Field(
        default=300, description="Rate limit ceiling per minute for this tenant"
    )
    created_at: str = Field(description="ISO timestamp of account creation")
    authorized_at: str | None = Field(default=None, description="ISO timestamp of admin approval")
    authorized_by: str | None = Field(
        default=None, description="Admin identifier who approved access"
    )
    notes: str | None = Field(default=None, description="Commercial or compliance notes")


class EnterpriseCreateRequest(BaseModel):
    """Payload to register a new enterprise client."""

    name: str = Field(min_length=2, description="Enterprise company name")
    contact_email: str = Field(min_length=5, description="Primary contact email")
    plan: str = Field(default="growth", description="Plan tier: trial, growth, enterprise")
    rate_limit_per_minute: int = Field(default=300, ge=10, le=10000)
    notes: str | None = None


class EnterpriseAuthorizeRequest(BaseModel):
    """Payload for admin authorizing an enterprise for production."""

    admin_user: str = Field(default="admin@sentinel.internal", description="Admin name or email")
    rate_limit_per_minute: int | None = Field(
        default=None, description="Optional updated rate limit"
    )
    notes: str | None = None


def generate_key(prefix: str) -> str:
    """Generate a high-entropy API key with given prefix."""
    random_part = secrets.token_hex(16)
    return f"{prefix}_{random_part}"


class EnterpriseManager:
    """Manages enterprise client records, API key authentication, and authorization."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url
        self._redis_client: Any = None
        # Thread-safe in-memory store for fallback / standalone / testing
        self._enterprises: dict[str, Enterprise] = {}
        # Key -> (enterprise_id, environment)
        self._key_index: dict[str, tuple[str, str]] = {}
        self._seed_default_enterprises()

    def _seed_default_enterprises(self) -> None:
        """Seed realistic enterprise accounts on initialization."""
        e1 = Enterprise(
            id="ent_acme_financial",
            name="Acme Financial Systems",
            contact_email="eng-leads@acmefinancial.com",
            plan="enterprise",
            status="authorized",
            test_api_key="sk_test_acme_9f81a74e2d3b4c5a19283746",
            live_api_key="sk_live_acme_6e20c81b9f4a3d7e50291827",
            test_allowed=True,
            production_allowed=True,
            test_requests_count=142,
            live_requests_count=12850,
            rate_limit_per_minute=1200,
            created_at=datetime.now(UTC).isoformat(),
            authorized_at=datetime.now(UTC).isoformat(),
            authorized_by="admin@sentinel.internal",
            notes="Enterprise SLA, custom Qdrant dedicated collection, SOC-2 compliant",
        )
        e2 = Enterprise(
            id="ent_nova_health",
            name="Nova Health AI",
            contact_email="dev@novahealth.io",
            plan="growth",
            status="pending",
            test_api_key="sk_test_nova_3b14d89a7c2e5f1a94837261",
            live_api_key="sk_live_nova_8d71e24c5b3a9f0e12345678",
            test_allowed=True,
            production_allowed=False,
            test_requests_count=38,
            live_requests_count=0,
            rate_limit_per_minute=300,
            created_at=datetime.now(UTC).isoformat(),
            authorized_at=None,
            authorized_by=None,
            notes="Applied for Production access. Integration tests verified, pending HIPAA BAA review.",
        )
        e3 = Enterprise(
            id="ent_cloudscale_logistics",
            name="CloudScale Logistics",
            contact_email="integrations@cloudscale.net",
            plan="trial",
            status="pending",
            test_api_key="sk_test_scale_7a2f9c4d1e8b3a0e99887766",
            live_api_key="sk_live_scale_1f4c7e9a2d8b5c3d44332211",
            test_allowed=True,
            production_allowed=False,
            test_requests_count=12,
            live_requests_count=0,
            rate_limit_per_minute=60,
            created_at=datetime.now(UTC).isoformat(),
            authorized_at=None,
            authorized_by=None,
            notes="Testing Node.js SDK in sandbox. Ready for production approval after pilot.",
        )

        for ent in [e1, e2, e3]:
            self._save_to_memory(ent)

    def _save_to_memory(self, enterprise: Enterprise) -> None:
        self._enterprises[enterprise.id] = enterprise
        self._key_index[enterprise.test_api_key] = (enterprise.id, "test")
        self._key_index[enterprise.live_api_key] = (enterprise.id, "production")

    async def list_enterprises(self) -> list[Enterprise]:
        """Return all registered enterprises."""
        return list(self._enterprises.values())

    async def get_enterprise(self, enterprise_id: str) -> Enterprise | None:
        """Retrieve an enterprise by ID."""
        return self._enterprises.get(enterprise_id)

    async def create_enterprise(self, req: EnterpriseCreateRequest) -> Enterprise:
        """Register a new enterprise client with both test and production keys."""
        ent_id = f"ent_{secrets.token_hex(6)}"
        test_key = generate_key("sk_test")
        live_key = generate_key("sk_live")

        enterprise = Enterprise(
            id=ent_id,
            name=req.name,
            contact_email=req.contact_email,
            plan=req.plan,
            status="pending",
            test_api_key=test_key,
            live_api_key=live_key,
            test_allowed=True,
            production_allowed=False,
            test_requests_count=0,
            live_requests_count=0,
            rate_limit_per_minute=req.rate_limit_per_minute,
            created_at=datetime.now(UTC).isoformat(),
            authorized_at=None,
            authorized_by=None,
            notes=req.notes,
        )
        self._save_to_memory(enterprise)
        logger.info(
            "Registered new enterprise client",
            extra={"enterprise_id": ent_id, "enterprise_name": req.name, "plan": req.plan},
        )
        return enterprise

    async def authorize_enterprise(
        self,
        enterprise_id: str,
        admin_user: str = "admin@sentinel.internal",
        rate_limit: int | None = None,
        notes: str | None = None,
    ) -> Enterprise | None:
        """Admin approves an enterprise for live production traffic."""
        ent = self._enterprises.get(enterprise_id)
        if not ent:
            return None

        ent.status = "authorized"
        ent.production_allowed = True
        ent.authorized_at = datetime.now(UTC).isoformat()
        ent.authorized_by = admin_user
        if rate_limit is not None:
            ent.rate_limit_per_minute = rate_limit
        if notes:
            ent.notes = notes

        self._save_to_memory(ent)
        logger.info(
            "Enterprise authorized for production access",
            extra={"enterprise_id": enterprise_id, "admin": admin_user},
        )
        return ent

    async def revoke_enterprise(
        self,
        enterprise_id: str,
        admin_user: str = "admin@sentinel.internal",
        reason: str | None = None,
    ) -> Enterprise | None:
        """Admin suspends or revokes enterprise production access."""
        ent = self._enterprises.get(enterprise_id)
        if not ent:
            return None

        ent.status = "suspended"
        ent.production_allowed = False
        ent.notes = f"Suspended by {admin_user}: {reason or 'Administrative review'}"
        self._save_to_memory(ent)
        logger.warning(
            "Enterprise production access suspended",
            extra={"enterprise_id": enterprise_id, "admin": admin_user},
        )
        return ent

    async def rotate_key(
        self,
        enterprise_id: str,
        environment: str = "test",
    ) -> tuple[Enterprise | None, str | None]:
        """Rotate either test or live API key for an enterprise."""
        ent = self._enterprises.get(enterprise_id)
        if not ent:
            return None, None

        if environment == "test":
            # Remove old key index
            self._key_index.pop(ent.test_api_key, None)
            new_key = generate_key("sk_test")
            ent.test_api_key = new_key
        else:
            self._key_index.pop(ent.live_api_key, None)
            new_key = generate_key("sk_live")
            ent.live_api_key = new_key

        self._save_to_memory(ent)
        logger.info(
            "Rotated enterprise API key",
            extra={"enterprise_id": enterprise_id, "environment": environment},
        )
        return ent, new_key

    def validate_key(self, api_key: str) -> tuple[bool, Enterprise | None, str, bool, str]:
        """Validate an incoming API key.

        Returns:
            (is_valid, enterprise, environment, is_authorized, error_msg)
        """
        match = self._key_index.get(api_key)
        if not match:
            return False, None, "unknown", False, "Invalid API key."

        ent_id, env = match
        ent = self._enterprises.get(ent_id)
        if not ent:
            return False, None, env, False, "Enterprise record not found."

        if env == "production":
            if not ent.production_allowed or ent.status != "authorized":
                return (
                    True,
                    ent,
                    env,
                    False,
                    f"Enterprise '{ent.name}' is pending production authorization. "
                    "An administrator must authorize your account before live traffic can be served.",
                )
            return True, ent, env, True, ""
        else:
            # Test / Sandbox
            if not ent.test_allowed or ent.status == "suspended":
                return (
                    True,
                    ent,
                    env,
                    False,
                    f"Enterprise '{ent.name}' test environment is currently suspended.",
                )
            return True, ent, env, True, ""

    def record_usage(self, enterprise_id: str, environment: str) -> None:
        """Increment request counters for telemetry and billing."""
        ent = self._enterprises.get(enterprise_id)
        if not ent:
            return
        if environment == "production":
            ent.live_requests_count += 1
        else:
            ent.test_requests_count += 1
