"""
app.py

Flask entrypoint for the Smart Stadium Fan Assistant.

Endpoints:
  GET  /                -> serves the fan-facing chat assistant
  GET  /ops              -> serves the staff-facing operations dashboard
  POST /api/chat        -> {message, history} -> {reply}
  GET  /api/status      -> raw live crowd/gate data (drives both / ribbon and /ops)
  GET  /api/health      -> simple health check (no API key required)
"""

import os
import time
from collections import defaultdict
from threading import Lock
from time import strftime
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional

from assistant import StadiumAssistant, get_live_status

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Configure CORS based on allowed origins environment variable
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "").strip()
if allowed_origins_str:
    origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins}})
else:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

# Token-bucket rate limiter for API protection
class TokenBucket:
    def __init__(self, capacity=30, refill_rate=0.5):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def consume(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

IP_LIMITS = defaultdict(lambda: TokenBucket(30, 0.5))

@app.before_request
def rate_limit():
    if request.path == "/api/chat":
        ip = request.remote_addr or "unknown"
        if not IP_LIMITS[ip].consume():
            res = jsonify({"error": "rate_limit_exceeded", "detail": "Too many requests. Please try again later."})
            res.headers["Retry-After"] = "2"
            return res, 429

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:;"
    )
    return response

# Pydantic Input Validation Models
class ChatHistoryItem(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError("Role must be 'user' or 'assistant'.")
        return v

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatHistoryItem] = Field(default_factory=list)
    accessibility_profile: Optional[str] = None

    @field_validator("accessibility_profile")
    @classmethod
    def validate_profile(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"none", "wheelchair", "sensory", "companion", "asl", "walking"}:
            raise ValueError("Invalid accessibility profile name.")
        return v

class IncidentRequest(BaseModel):
    action: str
    gate: Optional[str] = None
    enabled: Optional[bool] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    from_gate: Optional[str] = None
    to_gate: Optional[str] = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = {
            "close_gate", "reopen_gate", "set_transport_delay", 
            "set_gate_surge", "set_alert_threshold", 
            "deploy_volunteers", "clear_volunteers", "reset_simulation"
        }
        if v not in valid_actions:
            raise ValueError(f"Unsupported incident action: {v}")
        return v

def format_validation_error(err: ValidationError):
    clean_errors = []
    for e in err.errors():
        clean_err = {
            "loc": list(e.get("loc", [])),
            "msg": e.get("msg"),
            "type": e.get("type"),
        }
        if "input" in e:
            try:
                import json
                json.dumps(e["input"])
                clean_err["input"] = e["input"]
            except Exception:
                clean_err["input"] = str(e["input"])
        clean_errors.append(clean_err)
    return jsonify({"error": "invalid_parameters", "detail": clean_errors}), 400

_assistant = None
SUPPORTED_PROVIDERS = {"demo", "anthropic", "openai", "gemini", "openai_compatible"}
LOCAL_ENDPOINT_HOSTS = {"localhost", "127.0.0.1", "::1"}
OPS_STATE = {
    "closed_gates": set(),
    "transport_delay": False,
    "last_recommendation": None,
    "change_log": [],
    "alert_threshold": 0.75,
    "surged_gates": {},
    "volunteers_active": {},
    "congested_gates_tracked": []
}
OPS_LOCK = Lock()


def get_assistant() -> StadiumAssistant:
    global _assistant
    if _assistant is None:
        _assistant = StadiumAssistant()
    return _assistant


def get_server_ai_config() -> dict:
    """Read provider settings only from server environment variables.

    API credentials and endpoints must never come from a browser request. This
    keeps the service from becoming an unauthenticated API proxy.
    """
    provider = os.environ.get("AI_PROVIDER", "").strip().lower()
    if not provider:
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "demo"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Server AI_PROVIDER is unsupported.")

    key_variables = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai_compatible": "AI_API_KEY",
    }
    key_variable = key_variables.get(provider)
    api_key = os.environ.get(key_variable) if key_variable else None
    model = os.environ.get("AI_MODEL", "").strip() or None
    base_url = os.environ.get("AI_BASE_URL", "").strip() or None

    if provider == "openai_compatible":
        if not base_url:
            raise ValueError("Server AI_BASE_URL is required for openai_compatible.")
        parsed = urlparse(base_url)
        is_local_http = parsed.scheme == "http" and parsed.hostname in LOCAL_ENDPOINT_HOSTS
        if parsed.scheme != "https" and not is_local_http:
            raise ValueError("AI_BASE_URL must use HTTPS, except for localhost development.")
        if parsed.query or parsed.fragment or (parsed.path not in ("", "/")):
            raise ValueError("AI_BASE_URL must be an origin without a path, query, or fragment.")

    return {"provider": provider, "api_key": api_key, "model": model, "base_url": base_url}


def get_ops_status() -> dict:
    """Combine live crowd data with staff-controlled demo incidents."""
    with OPS_LOCK:
        closed_gates = set(OPS_STATE["closed_gates"])
        transport_delay = OPS_STATE["transport_delay"]
        surged_gates = dict(OPS_STATE["surged_gates"])
        volunteers_active = dict(OPS_STATE["volunteers_active"])
        alert_threshold = OPS_STATE["alert_threshold"]

    status_data = get_live_status(
        closed_gates,
        transport_delay,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
        alert_threshold=alert_threshold,
    )
    recommendation = status_data["recommended_gate"]
    
    with OPS_LOCK:
        # Check threshold crossings statefully to log alerts only once
        current_congested = set()
        for g in status_data["gates"]:
            if g["score"] >= alert_threshold and not g["closed"]:
                current_congested.add(g["name"])
        
        previous_congested = set(OPS_STATE.get("congested_gates_tracked", []))
        new_congested = current_congested - previous_congested
        resolved_congested = previous_congested - current_congested - closed_gates
        
        for g_name in new_congested:
            score_pct = round(status_data["crowd"][g_name]["score"] * 100)
            threshold_pct = round(alert_threshold * 100)
            OPS_STATE["change_log"].insert(0, {
                "time": strftime("%H:%M:%S"),
                "type": "ALERT",
                "message": f"High Congestion: {g_name} crossed alert threshold ({score_pct}% >= {threshold_pct}%)."
            })
            
        for g_name in resolved_congested:
            score_pct = round(status_data["crowd"][g_name]["score"] * 100)
            threshold_pct = round(alert_threshold * 100)
            OPS_STATE["change_log"].insert(0, {
                "time": strftime("%H:%M:%S"),
                "type": "ALERT",
                "message": f"Congestion Eased: {g_name} congestion dropped below threshold ({score_pct}% < {threshold_pct}%)."
            })
            
        OPS_STATE["congested_gates_tracked"] = list(current_congested)

        # Recommendation change logging
        if recommendation != OPS_STATE["last_recommendation"]:
            previous = OPS_STATE["last_recommendation"]
            if previous is not None:
                explanation = status_data["recommendation_reason"]
                OPS_STATE["change_log"].insert(0, {
                    "time": strftime("%H:%M:%S"),
                    "type": "ROUTE_RECOM",
                    "message": f"Recommendation shifted to {recommendation or 'Escalate'}. Reason: {explanation}",
                })
            OPS_STATE["last_recommendation"] = recommendation
            
        # Truncate change log to last 15 items
        del OPS_STATE["change_log"][15:]
        
        status_data["change_log"] = OPS_STATE["change_log"]
        status_data["closed_gates"] = sorted(OPS_STATE["closed_gates"])
        status_data["surged_gates"] = OPS_STATE["surged_gates"]
        status_data["volunteers_active"] = OPS_STATE["volunteers_active"]
        
        try:
            config = get_server_ai_config()
            status_data["ai_provider"] = config.get("provider", "demo")
            status_data["ai_model"] = config.get("model", "default")
        except Exception:
            status_data["ai_provider"] = "demo"
            status_data["ai_model"] = "default"
            
    return status_data


def record_ops_change(message: str, category: str = "SYSTEM") -> None:
    with OPS_LOCK:
        OPS_STATE["change_log"].insert(0, {"time": strftime("%H:%M:%S"), "type": category, "message": message})
        del OPS_STATE["change_log"][15:]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    return jsonify(get_ops_status())


@app.route("/ops")
def ops_dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/ops/incidents", methods=["POST"])
def update_incident():
    """Demo-only staff control for closures and transport disruption exercises."""
    payload = request.get_json(force=True, silent=True) or {}
    try:
        req = IncidentRequest(**payload)
    except ValidationError as err:
        return format_validation_error(err)

    action = req.action
    gate = req.gate
    valid_gates = set(get_live_status()["crowd"])

    if action in {"close_gate", "reopen_gate"}:
        if gate not in valid_gates:
            return jsonify({"error": "valid gate is required"}), 400
        with OPS_LOCK:
            if action == "close_gate":
                OPS_STATE["closed_gates"].add(gate)
                # Also clear override and volunteers for this gate since it is closed now
                OPS_STATE["surged_gates"].pop(gate, None)
                OPS_STATE["volunteers_active"].pop(gate, None)
                message = f"Gate Closed: {gate} is now closed due to an operational issue."
            else:
                OPS_STATE["closed_gates"].discard(gate)
                message = f"Gate Reopened: {gate} has been reopened."
        record_ops_change(message, "INCIDENT")
        
    elif action == "set_transport_delay":
        enabled = bool(req.enabled)
        with OPS_LOCK:
            OPS_STATE["transport_delay"] = enabled
        record_ops_change(
            "Transport Delay Mode Activated: Advising rail fallbacks." if enabled 
            else "Transport Delay Mode Cleared: Transport returned to normal operations.",
            "INCIDENT"
        )
        
    elif action == "set_gate_surge":
        score = req.score
        if gate not in valid_gates:
            return jsonify({"error": "valid gate is required"}), 400
        with OPS_LOCK:
            if score is None:
                OPS_STATE["surged_gates"].pop(gate, None)
                message = f"Surge Cleared: {gate} congestion surge override removed."
            else:
                score_val = max(0.0, min(1.0, score))
                OPS_STATE["surged_gates"][gate] = score_val
                message = f"Surge Simulated: {gate} set to {round(score_val * 100)}% capacity."
        record_ops_change(message, "INCIDENT")
        
    elif action == "set_alert_threshold":
        threshold = req.threshold
        if threshold is None:
            return jsonify({"error": "threshold is required"}), 400
        threshold_val = max(0.1, min(0.99, threshold))
        with OPS_LOCK:
            OPS_STATE["alert_threshold"] = threshold_val
            message = f"Alert Threshold Shifted: Operator updated alert limit to {round(threshold_val * 100)}%."
        record_ops_change(message, "STAFF_ACTION")
        
    elif action == "deploy_volunteers":
        from_gate = req.from_gate
        to_gate = req.to_gate
        if from_gate not in valid_gates or to_gate not in valid_gates:
            return jsonify({"error": "valid gates required"}), 400
        with OPS_LOCK:
            OPS_STATE["volunteers_active"][from_gate] = True
            message = f"Volunteers Routed: Dispatched staff from {from_gate} to {to_gate} to optimize flow."
        record_ops_change(message, "STAFF_ACTION")
        
    elif action == "clear_volunteers":
        if gate not in valid_gates:
            return jsonify({"error": "valid gate is required"}), 400
        with OPS_LOCK:
            OPS_STATE["volunteers_active"].pop(gate, None)
            message = f"Staff Reset: Volunteer dispatch cleared at {gate}."
        record_ops_change(message, "STAFF_ACTION")
        
    elif action == "reset_simulation":
        with OPS_LOCK:
            OPS_STATE["closed_gates"].clear()
            OPS_STATE["transport_delay"] = False
            OPS_STATE["surged_gates"].clear()
            OPS_STATE["volunteers_active"].clear()
            OPS_STATE["alert_threshold"] = 0.75
            OPS_STATE["last_recommendation"] = None
            OPS_STATE["change_log"] = []
            OPS_STATE["congested_gates_tracked"] = []
            message = "Simulation Reset: Cleared all operational overrides and logs."
        record_ops_change(message, "SYSTEM")
        
    else:
        return jsonify({"error": "unsupported incident action"}), 400

    return jsonify(get_ops_status())


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        req = ChatRequest(**payload)
    except ValidationError as err:
        return format_validation_error(err)

    message = req.message.strip()
    clean_history = [
        {"role": h.role, "content": h.content}
        for h in req.history
    ][-10:]

    accessibility_profile = req.accessibility_profile
    if accessibility_profile == "none":
        accessibility_profile = None

    try:
        config = get_server_ai_config()
        with OPS_LOCK:
            closed_gates = set(OPS_STATE["closed_gates"])
            transport_delay = OPS_STATE["transport_delay"]
            surged_gates = dict(OPS_STATE["surged_gates"])
            volunteers_active = dict(OPS_STATE["volunteers_active"])
            
        reply = get_assistant().ask(
            message, 
            clean_history, 
            closed_gates=closed_gates,
            transport_delay=transport_delay,
            surged_gates=surged_gates,
            volunteers_active=volunteers_active,
            accessibility_profile=accessibility_profile,
            **config
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # Do not expose provider responses, which can contain sensitive details.
        return jsonify({"error": "assistant_failed", "detail": "The selected provider request failed."}), 502

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
