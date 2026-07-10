"""
crowd_sim.py

Simulates real-time crowd density at each stadium gate.

In a production deployment this module would be replaced with a real feed
(e.g. turnstile counters, CCTV-based people-counting, or a stadium IoT
platform). For this demo/hackathon build, it deterministically generates a
plausible, time-varying congestion figure per gate so the assistant can
reason about *current* conditions rather than only static facts.

Kept dependency-free (stdlib only) to keep the repo small and portable.
"""

import math
import time
from typing import Dict


def _time_wave(seed: float, period_seconds: int = 900, now: float | None = None) -> float:
    """Smooth 0..1 oscillation used to simulate crowds building pre-kickoff
    and thinning post-kickoff, without needing an external data source."""
    t = (time.time() if now is None else now) + seed
    return (math.sin(2 * math.pi * (t % period_seconds) / period_seconds) + 1) / 2


def _score_for_gate(
    base: float,
    seed: float,
    now: float | None = None,
    gate_name: str | None = None,
    surged_gates: Dict[str, float] | None = None,
    volunteers_active: Dict[str, bool] | None = None,
) -> float:
    if surged_gates and gate_name in surged_gates:
        score = surged_gates[gate_name]
    else:
        wave = _time_wave(seed=seed, now=now)
        score = base + (wave - 0.5) * 0.6

    if volunteers_active and volunteers_active.get(gate_name):
        score -= 0.20

    return max(0.0, min(1.0, score))


def get_live_crowd_levels(
    baseline: Dict[str, float],
    now: float | None = None,
    surged_gates: Dict[str, float] | None = None,
    volunteers_active: Dict[str, bool] | None = None,
) -> Dict[str, dict]:
    """Return a congestion score (0.0 = empty, 1.0 = at capacity) and a
    human-readable label for every gate in `baseline`.
    """
    levels = {}
    for i, (gate, base) in enumerate(baseline.items()):
        score = _score_for_gate(
            base,
            seed=i * 137.0,
            now=now,
            gate_name=gate,
            surged_gates=surged_gates,
            volunteers_active=volunteers_active,
        )
        levels[gate] = {
            "score": round(score, 2),
            "label": _label_for_score(score),
        }
    return levels


def get_crowd_trends(
    baseline: Dict[str, float],
    minutes: int = 30,
    points: int = 7,
    surged_gates: Dict[str, float] | None = None,
    volunteers_active: Dict[str, bool] | None = None,
) -> Dict[str, list[float]]:
    """Return a deterministic rolling history for the staff operations view.

    Production would read this from stored turnstile/CCTV measurements. The
    demo recomputes historical points from the same time wave as live status.
    """
    now = time.time()
    offsets = [-(minutes * 60) + i * (minutes * 60 / (points - 1)) for i in range(points)]
    
    trends = {}
    for i, (gate, base) in enumerate(baseline.items()):
        gate_trend = []
        for offset in offsets:
            score = _score_for_gate(
                base,
                seed=i * 137.0,
                now=now + offset,
                gate_name=gate,
            )
            gate_trend.append(score)

        current_score = _score_for_gate(
            base,
            seed=i * 137.0,
            now=now,
            gate_name=gate,
            surged_gates=surged_gates,
            volunteers_active=volunteers_active,
        )
        original_now = gate_trend[-1]
        diff = current_score - original_now
        for idx in range(len(gate_trend)):
            fraction = idx / (len(gate_trend) - 1) if len(gate_trend) > 1 else 1.0
            gate_trend[idx] = max(0.0, min(1.0, round(gate_trend[idx] + diff * fraction, 2)))

        trends[gate] = gate_trend
    return trends


def _label_for_score(score: float) -> str:
    if score < 0.25:
        return "light"
    if score < 0.5:
        return "moderate"
    if score < 0.75:
        return "busy"
    return "very busy"


def recommend_gate(
    baseline: Dict[str, float],
    accessible_only: bool = False,
    gates_meta=None,
    excluded_gates: set[str] | None = None,
    surged_gates: Dict[str, float] | None = None,
    volunteers_active: Dict[str, bool] | None = None,
) -> str | None:
    """Pick the least-congested gate, optionally filtered to accessible gates.

    `baseline` keys look like "Gate A"; `gates_meta` entries have an
    "id" like "A" and an "accessible" bool, so we map name -> id to filter.
    """
    levels = get_live_crowd_levels(
        baseline,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )
    candidates = list(levels.items())
    if excluded_gates:
        candidates = [(name, info) for name, info in candidates if name not in excluded_gates]

    if accessible_only and gates_meta:
        accessible_ids = {g["id"] for g in gates_meta if g.get("accessible")}
        candidates = [(name, v) for name, v in candidates if name.split()[-1] in accessible_ids]

    if not candidates:
        return None
    best_gate = min(candidates, key=lambda kv: kv[1]["score"])
    return best_gate[0]

