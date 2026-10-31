"""Named custom auxiliary clients must follow the pool used by the main client."""

import json

import pytest

from agent import auxiliary_client as aux
from hermes_cli.auth import write_credential_pool


@pytest.mark.parametrize("pool_key", ["judge", "custom:judge"])
def test_custom_client_follows_pool_rotation(tmp_path, monkeypatch, pool_key):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "model": {"provider": "custom:judge", "default": "example-model"},
        "providers": {"judge": {"name": "Judge", "base_url": "https://example.invalid/v1", "key_env": ""}},
    }))
    rows = [{"id": str(i), "label": f"key-{i}", "auth_type": "api_key", "priority": i,
             "source": "manual", "access_token": f"test-key-{i}",
             "base_url": "https://example.invalid/v1"} for i in range(2)]
    write_credential_pool(pool_key, rows)
    before, _ = aux._get_cached_client("custom:judge", "example-model")
    # The primary agent records exhaustion in the same persistent pool.
    from agent.credential_pool import load_pool
    pool = load_pool(pool_key)
    replacement = pool.mark_exhausted_and_rotate(
        status_code=429, error_context={"message": "rate limit"}, api_key_hint=before.api_key,
    )
    after, _ = aux._get_cached_client("custom:judge", "example-model")
    assert after.api_key == replacement.runtime_api_key != before.api_key

    other = tmp_path / "other-profile"
    other.mkdir()
    (other / "config.yaml").write_text((tmp_path / "config.yaml").read_text())
    monkeypatch.setenv("HERMES_HOME", str(other))
    write_credential_pool(pool_key, [{**rows[0], "access_token": "other-profile-key"}])
    isolated, _ = aux._get_cached_client("custom:judge", "example-model")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    returned, _ = aux._get_cached_client("custom:judge", "example-model")
    assert isolated.api_key == "other-profile-key" and returned.api_key == after.api_key


def test_subscription_rejection_rotates_named_custom_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "providers": {"judge": {"name": "Judge", "base_url": "https://example.invalid/v1", "key_env": ""}},
    }))
    write_credential_pool("custom:judge", [
        {"id": str(i), "label": f"key-{i}", "auth_type": "api_key", "priority": i,
         "source": "manual", "access_token": f"test-key-{i}"} for i in range(2)
    ])
    class SubscriptionError(Exception):
        status_code = 403
    error = SubscriptionError("An active OpenCode Go subscription is required to use Go models.")
    assert aux._recover_provider_pool("custom:judge", error, failed_api_key="test-key-0")
    assert aux._peek_pool_entry("judge").runtime_api_key == "test-key-1"
