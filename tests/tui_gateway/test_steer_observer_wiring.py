"""TUI-side single rendering owner for accepted steers.

Mirrors tests/hermes_cli/test_steer_observer_wiring.py: when agent.steer()
accepts text in a TUI-gateway session, attached TUI clients must render one
'Steer queued' notice — via the acceptance observer wired by
``server._wire_session_agent`` (the gateway counterpart of the CLI's
``_init_agent`` wiring). Delivery semantics are untouched: only presentation
is asserted here, plus the no-duplicate rule for the local /steer path.

No model request is made.
"""
from __future__ import annotations

import threading
import types

import pytest

from run_agent import AIAgent
from tui_gateway import server

SID = "tui-steer-wiring-sid"


def _stub_agent() -> AIAgent:
    """AIAgent with only the steer/clear_interrupt state installed — mirrors
    the object.__new__ stub pattern used by the CLI wiring tests."""
    agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    return agent


@pytest.fixture
def wired_session(monkeypatch):
    """Live session record + agent wired by the production path.

    The fixture must NOT install _steer_accepted_callback itself:
    server._wire_session_agent owns it.
    """
    emitted = []
    monkeypatch.setattr(
        server, "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )
    agent = _stub_agent()
    assert not callable(getattr(agent, "_steer_accepted_callback", None))
    server._sessions[SID] = {
        "agent": agent,
        "session_key": "tui-steer-wiring-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "running": True,
        "last_active": 0.0,
    }
    server._wire_session_agent(SID, "tui-steer-wiring-key", agent)
    assert callable(getattr(agent, "_steer_accepted_callback", None))
    emitted.clear()
    yield agent, emitted
    server._sessions.pop(SID, None)


def _notices(emitted):
    return [
        payload for event, _sid, payload in emitted
        if event == "status.update"
        and isinstance(payload, dict)
        and "Steer queued:" in str(payload.get("text", ""))
    ]


def _all_queued_mentions(agent_and_emitted, extra_texts=()):
    _agent, emitted = agent_and_emitted
    mentions = sum(
        str(payload).count("Steer queued")
        for _event, _sid, payload in emitted
    )
    return mentions + sum(t.count("Steer queued") for t in extra_texts)


def test_external_acceptance_is_rendered_by_session_wiring(wired_session):
    agent, emitted = wired_session
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(agent.steer("  focus on errors  ")))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert outcomes == [True]
    assert agent._pending_steer == "focus on errors"
    notices = _notices(emitted)
    assert len(notices) == 1, "accepted external steer must render through _wire_session_agent"
    assert "focus on errors" in notices[0]["text"]


def test_local_slash_steer_is_not_echoed_twice(wired_session):
    agent, emitted = wired_session
    res = server._methods["command.dispatch"](
        "1", {"name": "steer", "arg": "local instruction", "session_id": SID})
    assert agent._pending_steer == "local instruction"
    result = res["result"]
    assert result["type"] == "exec"
    assert _all_queued_mentions(wired_session, (result.get("output", ""),)) == 1
    assert len(_notices(emitted)) == 1


def test_session_steer_rpc_broadcasts_single_notice(wired_session):
    agent, emitted = wired_session
    resp = server.handle_request(
        {"id": "1", "method": "session.steer",
         "params": {"session_id": SID, "text": "rpc guidance"}})
    assert resp["result"]["status"] == "queued"
    assert agent._pending_steer == "rpc guidance"
    notices = _notices(emitted)
    assert len(notices) == 1
    assert "rpc guidance" in notices[0]["text"]
    # The RPC reply itself carries no notice text: the broadcast owns rendering.
    assert "Steer queued" not in str(resp["result"])


def test_empty_and_internal_requeue_have_no_new_notice(wired_session):
    agent, emitted = wired_session
    assert agent.steer("  ") is False
    assert agent.steer("recovered instruction", _notify=False) is True
    assert agent._pending_steer == "recovered instruction"
    assert _notices(emitted) == []
    assert emitted == []


def test_soft_clear_preserves_accepted_steer_but_hard_cancel_drops_it(wired_session):
    agent, emitted = wired_session
    assert agent.steer("keep this across recovery") is True
    agent.clear_interrupt()
    assert agent._pending_steer == "keep this across recovery"
    assert len(_notices(emitted)) == 1
    agent.clear_interrupt(hard_cancel=True)
    assert agent._pending_steer is None
    assert len(_notices(emitted)) == 1


def test_unwired_agent_keeps_local_slash_notice(monkeypatch):
    """Fallback parity with the CLI: a bare agent without session wiring still
    answers /steer with a local notice instead of going silent."""
    emitted = []
    monkeypatch.setattr(
        server, "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )
    steered = []

    def _steer(text):
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        steered.append(cleaned)
        return True

    agent = types.SimpleNamespace(steer=_steer)
    assert not callable(getattr(agent, "_steer_accepted_callback", None))
    server._sessions[SID] = {
        "agent": agent,
        "session_key": "tui-steer-fallback-key",
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
    }
    try:
        res = server._methods["command.dispatch"](
            "1", {"name": "steer", "arg": "fallback note", "session_id": SID})
    finally:
        server._sessions.pop(SID, None)
    assert steered == ["fallback note"]
    output = res["result"].get("output", "")
    assert "Steer queued" in output
    assert _notices(emitted) == []
