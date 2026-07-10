"""
Unit tests for the Smart Stadium Assistant.

Run with:  pytest tests/ -v
(from the repo root, with backend/ on PYTHONPATH — see conftest.py)

These tests avoid calling the real Anthropic API (no network / no cost):
- crowd_sim tests exercise the deterministic simulation logic directly.
- app tests use Flask's test client and monkeypatch the assistant call.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from crowd_sim import get_crowd_trends, get_live_crowd_levels, recommend_gate, _label_for_score  # noqa: E402
from assistant import StadiumAssistant, get_live_status  # noqa: E402


BASELINE = {"Gate A": 0.4, "Gate B": 0.6, "Gate C": 0.3, "Gate D": 0.15}
GATES_META = [
    {"id": "A", "accessible": True},
    {"id": "B", "accessible": True},
    {"id": "C", "accessible": False},
    {"id": "D", "accessible": True},
]


def test_crowd_levels_are_bounded():
    levels = get_live_crowd_levels(BASELINE)
    assert set(levels.keys()) == set(BASELINE.keys())
    for gate, info in levels.items():
        assert 0.0 <= info["score"] <= 1.0
        assert info["label"] in ("light", "moderate", "busy", "very busy")


def test_label_thresholds():
    assert _label_for_score(0.0) == "light"
    assert _label_for_score(0.3) == "moderate"
    assert _label_for_score(0.6) == "busy"
    assert _label_for_score(0.9) == "very busy"


def test_recommend_gate_returns_valid_gate():
    gate = recommend_gate(BASELINE)
    assert gate in BASELINE


def test_recommend_gate_accessible_only_excludes_inaccessible():
    # Gate C is not accessible; run several times since the simulation is
    # time-varying and we want confidence it's never selected under this filter.
    for _ in range(5):
        gate = recommend_gate(BASELINE, accessible_only=True, gates_meta=GATES_META)
        assert gate != "Gate C"


def test_crowd_trends_return_30_minute_history():
    trends = get_crowd_trends(BASELINE)
    assert set(trends) == set(BASELINE)
    assert all(len(history) == 7 for history in trends.values())
    assert all(0.0 <= point <= 1.0 for history in trends.values() for point in history)


def test_gate_closure_recalculates_recommendation_and_records_status():
    normal_status = get_live_status()
    original_gate = normal_status["recommended_gate"]
    incident_status = get_live_status(closed_gates={original_gate}, transport_delay=True)
    assert incident_status["recommended_gate"] != original_gate
    assert incident_status["transport_delay"] is True
    assert any(gate["name"] == original_gate and gate["closed"] for gate in incident_status["gates"])
    assert any("closed" in alert for alert in incident_status["alerts"])


def test_stadium_data_loads_and_has_required_keys():
    data_path = Path(__file__).parent.parent / "backend" / "data" / "stadium.json"
    with open(data_path) as f:
        data = json.load(f)
    for key in ("stadium_name", "gates", "zones", "transport", "accessibility_services", "crowd_baseline"):
        assert key in data
    assert len(data["gates"]) > 0
    assert all("id" in g and "accessible" in g for g in data["gates"])


def test_demo_maps_sections_to_gates_and_amenities():
    reply = StadiumAssistant().ask("I am in section 215", provider="demo")
    assert "Club Level - Sections 200-239" in reply
    assert "Gate D" in reply
    assert "nursing room" in reply


def test_demo_finds_known_amenities_and_transport():
    assistant = StadiumAssistant()
    sensory_reply = assistant.ask("Where is the sensory room?", provider="demo")
    airport_reply = assistant.ask("I am leaving now for the airport", provider="demo")
    assert "Family Zone / Sensory Room" in sensory_reply
    assert "Green Shuttle - Airport Express" in airport_reply


def test_demo_explains_accessible_gate_recommendation():
    reply = StadiumAssistant().ask("I need a wheelchair entrance", provider="demo")
    assert "step-free" in reply
    assert "currently" in reply


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    import app as flask_app_module
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_status_endpoint_shape(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.get_json()
    assert "crowd" in body
    assert "recommended_gate" in body
    assert "gates" in body
    assert "alerts" in body
    assert isinstance(body["alerts"], list)
    for gate in body["gates"]:
        assert "accessible" in gate
        assert "score" in gate


def test_ops_dashboard_route_serves_html(client):
    res = client.get("/ops")
    assert res.status_code == 200
    assert b"Operations" in res.data


def test_chat_rejects_empty_message(client):
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 400


def test_chat_rejects_oversized_message(client):
    res = client.post("/api/chat", json={"message": "x" * 3000})
    assert res.status_code == 400


def test_chat_calls_assistant(client, monkeypatch):
    import app as flask_app_module

    class FakeAssistant:
        def ask(self, message, history, **config):
            return f"echo: {message}"

    monkeypatch.setattr(flask_app_module, "get_assistant", lambda: FakeAssistant())
    res = client.post("/api/chat", json={"message": "Where is Gate A?", "history": []})
    assert res.status_code == 200
    assert res.get_json()["reply"] == "echo: Where is Gate A?"


def test_chat_ignores_client_provider_and_key(client, monkeypatch):
    import app as flask_app_module

    received = {}

    class FakeAssistant:
        def ask(self, message, history, **config):
            received.update(config)
            return "safe reply"

    monkeypatch.setattr(flask_app_module, "get_assistant", lambda: FakeAssistant())
    res = client.post("/api/chat", json={
        "message": "Which gate?",
        "provider": "gemini",
        "api_key": "browser-key-must-not-be-used",
        "base_url": "http://internal-service",
    })
    assert res.status_code == 200
    assert received["provider"] == "anthropic"
    assert received["api_key"] == "test-key-not-used"
    assert received["base_url"] is None


def test_custom_endpoint_requires_https_or_localhost(monkeypatch):
    import app as flask_app_module

    monkeypatch.setenv("AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AI_API_KEY", "server-key")
    monkeypatch.setenv("AI_BASE_URL", "http://example.com")
    with pytest.raises(ValueError, match="HTTPS"):
        flask_app_module.get_server_ai_config()

    monkeypatch.setenv("AI_BASE_URL", "http://localhost:11434")
    assert flask_app_module.get_server_ai_config()["base_url"] == "http://localhost:11434"


def test_chat_works_without_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app as flask_app_module

    flask_app_module._assistant = None
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        res = client.post("/api/chat", json={"message": "Which accessible gate should I use?"})

    assert res.status_code == 200
    assert "Demo mode" in res.get_json()["reply"]


def test_custom_gate_surge_simulation():
    # Direct simulation test
    baseline = {"Gate A": 0.4, "Gate B": 0.6}
    levels = get_live_crowd_levels(baseline, surged_gates={"Gate B": 0.90})
    assert levels["Gate B"]["score"] == 0.90
    assert levels["Gate B"]["label"] == "very busy"
    
    # Smooth trend adjustment test
    trends = get_crowd_trends(baseline, surged_gates={"Gate B": 0.90})
    assert trends["Gate B"][-1] == 0.90
    # The first point should be closer to the baseline/original, showing a rising trend
    assert trends["Gate B"][0] < 0.90


def test_volunteer_routing_relief():
    from crowd_sim import _score_for_gate
    baseline = {"Gate A": 0.4, "Gate B": 0.6}
    # Standard score for Gate B surged to 0.90
    score_surged = _score_for_gate(0.6, seed=137.0, gate_name="Gate B", surged_gates={"Gate B": 0.90})
    assert score_surged == 0.90
    
    # Surged to 0.90 but volunteers active should apply a -0.20 relief factor
    score_relieved = _score_for_gate(0.6, seed=137.0, gate_name="Gate B", surged_gates={"Gate B": 0.90}, volunteers_active={"Gate B": True})
    assert score_relieved == 0.70


def test_custom_alert_threshold():
    # If alert threshold is 0.80, a gate at 0.76 should not trigger surge alert, but at 0.82 it should
    status = get_live_status(surged_gates={"Gate A": 0.76}, alert_threshold=0.80)
    assert not any("Gate Surge Alert: Gate A" in alert for alert in status["alerts"])
    
    status_high = get_live_status(surged_gates={"Gate A": 0.85}, alert_threshold=0.80)
    assert any("Gate Surge Alert: Gate A" in alert for alert in status_high["alerts"])


def test_incident_api_and_event_logging(client):
    # Reset simulation
    res = client.post("/api/ops/incidents", json={"action": "reset_simulation"})
    assert res.status_code == 200
    
    # Set Gate B surge to 0.90
    res = client.post("/api/ops/incidents", json={"action": "set_gate_surge", "gate": "Gate B", "score": 0.90})
    assert res.status_code == 200
    body = res.get_json()
    assert body["surged_gates"]["Gate B"] == 0.90
    
    # Verify log contains Surge Simulated event
    assert any(item["type"] == "INCIDENT" and "Gate B" in item["message"] and "Surge" in item["message"] for item in body["change_log"])
    
    # Change threshold to 0.85
    res = client.post("/api/ops/incidents", json={"action": "set_alert_threshold", "threshold": 0.85})
    assert res.status_code == 200
    body = res.get_json()
    assert body["alert_threshold"] == 0.85
    assert any(item["type"] == "STAFF_ACTION" and "Threshold" in item["message"] for item in body["change_log"])
    
    # Route volunteers from Gate B to Gate A
    res = client.post("/api/ops/incidents", json={"action": "deploy_volunteers", "from_gate": "Gate B", "to_gate": "Gate A"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["volunteers_active"]["Gate B"] is True
    assert any(item["type"] == "STAFF_ACTION" and "Volunteers Routed" in item["message"] for item in body["change_log"])


def test_assistant_incident_awareness():
    assistant = StadiumAssistant()
    
    # Ask about closed Gate B in demo mode
    reply_closed = assistant.ask("Is Gate B open?", provider="demo", closed_gates={"Gate B"})
    assert "Gate B is currently CLOSED" in reply_closed
    
    # Ask about transport during transport delay
    reply_delay = assistant.ask("How do I get to the airport?", provider="demo", transport_delay=True)
    assert "NOTICE: Active transport delay" in reply_delay or "Expect major shuttle delays" in reply_delay


def test_accessibility_profiles_demo_routing():
    assistant = StadiumAssistant()
    
    # 1. Wheelchair Profile
    reply_wheelchair = assistant.ask(
        "Where should I enter?", 
        provider="demo", 
        accessibility_profile="wheelchair"
    )
    assert "Wheelchair Profile Active" in reply_wheelchair
    assert "step-free" in reply_wheelchair
    
    # 2. Sensory Profile
    reply_sensory = assistant.ask(
        "Is there a quiet room?", 
        provider="demo", 
        accessibility_profile="sensory"
    )
    assert "Sensory-Friendly Profile Active" in reply_sensory
    assert "Quiet sensory room" in reply_sensory
    
    # 3. Short Distance Profile
    reply_walking = assistant.ask(
        "How do I get to rail station?", 
        provider="demo", 
        accessibility_profile="walking"
    )
    assert "Short Distance Profile Active" in reply_walking
    assert "Avoid Meadowlands Rail walk" in reply_walking


def test_accessibility_profile_chat_endpoint(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app as flask_app_module

    flask_app_module._assistant = None
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        res = client.post("/api/chat", json={
            "message": "Enter gate help",
            "accessibility_profile": "wheelchair"
        })

    assert res.status_code == 200
    body = res.get_json()
    assert "Wheelchair Profile Active" in body["reply"]


def test_no_key_demo_replies_comprehensive():
    assistant = StadiumAssistant()
    
    # 1. Accessibility query in demo mode
    reply = assistant.ask("I need a wheelchair route", provider="demo")
    assert "use Gate D" in reply or "use Gate A" in reply
    assert "step-free" in reply
    assert "Wheelchair rental" in reply
    
    # 2. Transport query in demo mode
    reply = assistant.ask("How do I take the shuttle downtown?", provider="demo")
    assert "Downtown Loop" in reply
    assert "shuttle" in reply.lower()
    
    # 3. Seating query in demo mode
    reply = assistant.ask("Where is Section 215?", provider="demo")
    assert "Section 215" in reply
    assert "Club Level" in reply
    
    # 4. Sensory-room query in demo mode
    reply = assistant.ask("Do you have a sensory quiet room?", provider="demo")
    assert "sensory room" in reply or "Sensory Room" in reply
    assert "Gate D" in reply


def test_extreme_congestion_rerouting():
    # If Gate A is extremely congested (95%) and Gate B is clear (10%),
    # Gate B should be recommended, and the alert list should contain a rerouting recommendation
    baseline = {"Gate A": 0.95, "Gate B": 0.10, "Gate C": 0.30, "Gate D": 0.40}
    status = get_live_status(
        surged_gates={"Gate A": 0.95, "Gate B": 0.10, "Gate C": 0.30, "Gate D": 0.40},
        alert_threshold=0.75
    )
    
    assert status["recommended_gate"] == "Gate B"
    # An alert should suggest moving volunteers from the surged gate (Gate A) to the recommended gate (Gate B)
    assert any("Suggested rerouting: Move volunteers from Gate A to Gate B" in alert for alert in status["alerts"])


def test_provider_selection_validation(monkeypatch):
    import app as flask_app_module
    
    monkeypatch.setenv("AI_PROVIDER", "invalid-provider")
    with pytest.raises(ValueError, match="unsupported"):
        flask_app_module.get_server_ai_config()


def test_reject_unsafe_custom_endpoints(monkeypatch):
    import app as flask_app_module
    monkeypatch.setenv("AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AI_API_KEY", "testkey")
    
    # 1. Reject paths
    monkeypatch.setenv("AI_BASE_URL", "https://example.com/v1")
    with pytest.raises(ValueError, match="without a path"):
        flask_app_module.get_server_ai_config()
        
    # 2. Reject queries
    monkeypatch.setenv("AI_BASE_URL", "https://example.com?key=value")
    with pytest.raises(ValueError, match="query"):
        flask_app_module.get_server_ai_config()
        
    # 3. Reject fragments
    monkeypatch.setenv("AI_BASE_URL", "https://example.com#fragment")
    with pytest.raises(ValueError, match="fragment"):
        flask_app_module.get_server_ai_config()


def test_api_key_never_appears_in_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app as flask_app_module
    
    class LeakKeyAssistant:
        def ask(self, *args, **kwargs):
            # Simulate a provider error containing the secret API key
            raise RuntimeError("Request failed with API key: secret_key_abc123")
            
    monkeypatch.setattr(flask_app_module, "get_assistant", lambda: LeakKeyAssistant())
    
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        res = client.post("/api/chat", json={"message": "Help"})
        
    assert res.status_code == 502
    body = res.get_json()
    assert "error" in body
    # Confirm the key was suppressed and does not appear in the response payload
    assert "secret_key_abc123" not in json.dumps(body)
    assert "assistant_failed" in body["error"]


def test_accessibility_users_never_routed_through_inaccessible_gates():
    baseline = {"Gate A": 0.4, "Gate B": 0.3, "Gate C": 0.1, "Gate D": 0.5}
    gates_meta = [
        {"id": "A", "accessible": True},
        {"id": "B", "accessible": True},
        {"id": "C", "accessible": False},
        {"id": "D", "accessible": True},
    ]
    
    # Normally Gate C would be recommended since it has lowest score (0.1).
    # But for accessible_only=True, it must be excluded!
    best_accessible = recommend_gate(
        baseline,
        accessible_only=True,
        gates_meta=gates_meta
    )
    assert best_accessible != "Gate C"
    assert best_accessible in {"Gate A", "Gate B", "Gate D"}
    
    # If all accessible gates are closed, it should return None rather than routing through Gate C
    best_accessible_none = recommend_gate(
        baseline,
        accessible_only=True,
        gates_meta=gates_meta,
        excluded_gates={"Gate A", "Gate B", "Gate D"}
    )
    assert best_accessible_none is None


def test_security_headers(client):
    res = client.get("/api/health")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert res.headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]


def test_rate_limiter(client, monkeypatch):
    import app as flask_app_module
    
    # Reset IP limits to verify clean rate state
    flask_app_module.IP_LIMITS.clear()
    
    class FakeAssistant:
        def ask(self, *args, **kwargs):
            return "ok"
    
    monkeypatch.setattr(flask_app_module, "get_assistant", lambda: FakeAssistant())
    
    # Send quick requests to deplete the bucket (capacity = 30 tokens)
    for _ in range(30):
        res = client.post("/api/chat", json={"message": "ping"})
        assert res.status_code == 200
        
    # The 31st request should be rate-limited
    res_limited = client.post("/api/chat", json={"message": "ping"})
    assert res_limited.status_code == 429
    assert res_limited.get_json()["error"] == "rate_limit_exceeded"
    assert "Retry-After" in res_limited.headers


def test_pydantic_validation(client):
    import app as flask_app_module
    # Clear rate limit bucket to prevent carry-over from test_rate_limiter
    flask_app_module.IP_LIMITS.clear()

    # 1. Reject invalid accessibility profiles
    res = client.post("/api/chat", json={"message": "ping", "accessibility_profile": "invalid-profile"})
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_parameters"
    
    # 2. Reject missing chat messages
    res2 = client.post("/api/chat", json={})
    assert res2.status_code == 400
    assert res2.get_json()["error"] == "invalid_parameters"
    
    # 3. Reject invalid history roles
    res3 = client.post("/api/chat", json={"message": "ping", "history": [{"role": "hacker", "content": "malicious"}]})
    assert res3.status_code == 400
    assert res3.get_json()["error"] == "invalid_parameters"

    # 4. Reject invalid operations console actions
    res4 = client.post("/api/ops/incidents", json={"action": "hacked_action"})
    assert res4.status_code == 400
    assert res4.get_json()["error"] == "invalid_parameters"


def test_cors_headers(client):
    # Test standard CORS settings (default is '*')
    res = client.get("/api/status")
    assert res.headers.get("Access-Control-Allow-Origin") == "*"




