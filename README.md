# Smart Stadium Fan Assistant — FIFA World Cup 2026

A GenAI-powered assistant that helps fans navigate a World Cup stadium in real time:
which gate to use, how to get to their seat, where the nearest accessible route or
sensory room is, and how to get home after the match — all grounded in live
(simulated) crowd data so the advice actually reflects conditions on the ground,
not just a static FAQ.

## Chosen vertical

**Smart Stadiums & Tournament Operations**, focused specifically on the **fan-facing
navigation, accessibility, and crowd-management** slice of that vertical.

## The problem this solves

At a World Cup-scale venue, the same three questions cause most of the friction:

1. *"Which gate should I use?"* — fans default to the gate they remember, even when
   it's the most congested one, creating bottlenecks and safety risk.
2. *"How do I get around if I have an accessibility need?"* — accessible routes,
   sensory rooms, and companion seating exist but are hard to discover in the moment.
3. *"How do I get home?"* — transport info is usually static signage, not tailored to
   when the fan is actually leaving or how they're traveling.

A generic chatbot that only answers FAQs doesn't solve this — it needs to reason
over the *current state* of the venue (crowd levels right now) combined with *who's
asking* (accessibility needs, language) to give a genuinely useful, real-time answer.

## Approach and logic

**Grounded generation, not free-form chat.** The LLM (Claude) never answers from its
own "knowledge" of stadiums. Every request is answered against two inputs injected
into the system prompt on every call:

1. **Static knowledge base** (`backend/data/stadium.json`) — gates, zones, amenities,
   transport options, accessibility services. This is the venue's source of truth and
   is trivial to swap for a real stadium's data.
2. **Live crowd data** (`backend/crowd_sim.py`) — a deterministic, time-varying
   simulation of gate congestion, standing in for a real feed (turnstile counters,
   CCTV people-counting, or a stadium IoT platform in production). It's recomputed on
   every request, so "which gate is fastest right now" is always current.

The system prompt explicitly instructs the model to **never invent facts** not
present in these two sources, and to say so and point to Guest Services if asked
something outside its data — this is the difference between a genuinely deployable
assistant and a demo that hallucinates gate numbers.

**Decision logic, not just retrieval.** `crowd_sim.recommend_gate()` actively picks
the least-congested gate, with an accessibility-aware filter so a wheelchair user is
never routed to a gate without step-free access — this is where "logical decision
making based on user context" shows up outside the LLM too, so the recommendation is
deterministic and testable independent of the model.

**Multilingual by default.** The system prompt instructs Claude to reply in whatever
language the fan writes in — no separate translation pipeline needed, since this is
something the model already does well when instructed clearly.

**Accessibility is proactive, not opt-in.** The system prompt tells the assistant to
surface relevant accessibility services even when not directly asked, because in
practice fans often don't know a service (like a sensory room or ASL interpretation)
exists at all.

## How the solution works

```
Fan (any language)
      │
      ▼
Flask app (backend/app.py) ── POST /api/chat
      │
      ▼
StadiumAssistant.ask()  (backend/assistant.py)
      │
      ├── loads backend/data/stadium.json          (static venue facts)
      ├── calls crowd_sim.get_live_crowd_levels()   (live congestion per gate)
      ├── calls crowd_sim.recommend_gate()          (best gate, incl. accessible-only)
      │
      ▼
Builds a grounded system prompt → calls Claude (Anthropic API)
      │
      ▼
Reply streamed back to the concourse-style chat UI (frontend/index.html)
```

A second endpoint, `GET /api/status`, exposes the raw live crowd numbers
independent of the LLM — this is what powers the scrolling gate-congestion ribbon
at the top of the fan UI, and also drives a dedicated **staff operations console**
at `/ops` (linked from the fan UI header). The ops console shows every gate's live
congestion as a bar, flags which gates are step-free accessible, and surfaces
threshold-based alerts (e.g. "Gate B is very busy — consider directing fans to Gate
D instead") — all without invoking the LLM, since staff need fast, deterministic
numbers rather than a conversational answer.

### Project structure

```
backend/
  app.py            Flask routes: chat, status, health
  assistant.py       Builds the grounded system prompt, calls Claude
  crowd_sim.py       Deterministic live-crowd simulation + gate recommendation
  data/stadium.json  Venue knowledge base (gates, zones, transport, accessibility)
  requirements.txt
frontend/
  index.html         Fan-facing chat UI (no build step, no framework)
  dashboard.html     Staff-facing ops console at /ops (live congestion + alerts)
tests/
  test_app.py         Unit tests for crowd logic + Flask API (10 tests, no network calls)
.env.example
.gitignore
```

## Running it locally

```bash
git clone <your-repo-url>
cd fifa-stadium-assistant
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env
# Optional: configure AI_PROVIDER, AI_MODEL, and the matching server-side key.
# See .env.example for Anthropic, OpenAI, Gemini, and OpenAI-compatible options.
# Without a key, the app runs in demo mode with deterministic, grounded advice
# from the local stadium data: section-to-gate routing, amenities (restrooms,
# first aid, nursing and sensory rooms), accessible entry, and destination-aware
# transport. Gate replies include their current crowd level and rationale.

cd backend
export $(grep -v '^#' ../.env | xargs)   # or use python-dotenv / your shell's env loading
python app.py
```

Then open `http://localhost:5000`.

## Demo walkthrough

You can evaluate the application's capabilities in **no-key mode**, which runs deterministically using local stadium data and a simulated crowd/transit environment. Follow these steps:

1. **Ask for the shortest gate**:
   - In the Fan Assistant chat, ask: *"Which gate has the shortest line right now?"*
   - The assistant recommends the least-congested gate (e.g. Gate D or Gate C) based on current simulated congestion scores.
2. **Demonstrate Accessibility Profiles**:
   - Click the **♿ Step-free / Wheelchair** profile chip at the top.
   - The Guidance Banner instantly displays, recommending **Gate D** (VIP/Accessible priority) with step-free concourse routes.
   - Ask the assistant: *"Where should I enter?"*. The assistant responds tailorig the answer to wheelchair routing.
3. **Trigger Operational Incidents & Real-time Rerouting**:
   - Open the Operations Console at `/ops` (`http://localhost:5000/ops`) in another window to view live congestion levels.
   - Click the **"Close Gate"** button next to **Gate D** (or trigger the *Gate B Closure & Redirect* preset).
   - Switch back to the Fan Assistant: the Guidance Banner *instantly* updates to steer wheelchair users to **Gate A** instead!
   - Ask the assistant: *"Is Gate D open?"*. The assistant (in deterministic demo mode) notes that Gate D is closed and redirects you to Gate A.
4. **Deploy Volunteers to Ease Surges**:
   - In the `/ops` console, simulate a congestion surge on **Gate B** (set it to 90%).
   - Observe the alert banner suggest: *"Suggested rerouting: Move volunteers from Gate B to Gate D."*
   - Click the **"Route Volunteers"** action button. Gate B's congestion score instantly drops by 20% due to staff deployment, and a `[STAFF_ACTION]` log is added to the timeline event viewer.
5. **Ask how to get home after the match**:
   - In the chat, ask: *"How do I get to the rail station?"* or *"What is the rideshare zone?"*.
   - The assistant outlines transport options (Meadowlands Rail Station is a 12-minute walk; rideshares pick up at Lot F).
   - If the **🚶 Short Distance** accessibility profile is active, it advises against Meadowlands Rail due to the 12-minute walk and recommends shuttle lines instead.
   - If **Activate transport delay** is triggered on the operations console, the assistant warns you about delay incidents on shuttle routes and suggests taking the rail station fallback.

### Running tests

```bash
pip install -r backend/requirements.txt
pytest tests/ -v
```

Tests cover the crowd-simulation logic (bounds, labeling, accessibility filtering)
and the Flask API (input validation, health/status endpoints, and the chat endpoint
with the Anthropic call mocked out — the test suite never makes a real network call
or requires an API key).

## Security & responsible-implementation notes

- The Anthropic API key is read only from an environment variable, never hardcoded,
  and `.env` is git-ignored.
- API provider, model, endpoint, and credentials are server-side environment
  configuration only; the browser never accepts, stores, or submits API keys.
- A custom OpenAI-compatible endpoint is administrator-configured only and must
  use HTTPS, except for explicit localhost development endpoints. This prevents
  the public app from being used as an arbitrary API proxy.
- `/api/chat` validates input (non-empty, length-capped) before it reaches the model,
  and caps how much conversation history is forwarded per request.
- The system prompt constrains the model to the provided data, reducing the risk of
  fabricated safety-relevant information (e.g. a wrong gate or wrong wait time).
- Errors from the Anthropic API are caught and returned as a generic `502` rather
  than leaking stack traces or internals to the client.
- No PII is collected or stored; the app is stateless per request (chat history is
  held client-side in the browser tab, not persisted server-side).

## Assumptions made

- **Live crowd data is simulated.** A real deployment would replace
  `crowd_sim.get_live_crowd_levels()` with a call to the stadium's actual
  people-counting or turnstile system — the rest of the app (grounding, prompt,
  UI) does not need to change, since it only depends on the `{gate: {score, label}}`
  shape returned by that function.
- **One demo venue.** `stadium.json` models a single representative stadium
  (structured after MetLife Stadium, a confirmed 2026 host venue) rather than all 16
  tournament venues, to keep the knowledge base focused and auditable within the
  scope of this challenge. The data model generalizes directly to other venues.
- **Model choice.** Uses `claude-sonnet-4-5` for response quality within the chat
  UI; swappable via the `MODEL` constant in `backend/assistant.py`.
- **No authentication layer.** Out of scope for this challenge; a production
  deployment would sit behind the venue's existing fan-app auth.

## Why this fits the evaluation criteria

- **Code quality** — small, single-responsibility modules (`app.py` / `assistant.py`
  / `crowd_sim.py`), no framework bloat, docstrings explaining intent.
- **Security** — see above: env-based secrets, input validation, no data leakage.
- **Efficiency** — stdlib-only simulation layer, single Flask process, no external
  services beyond the Anthropic API; frontend is one dependency-free HTML file.
- **Testing** — 10 unit tests covering both the deterministic decision logic and the
  API contract, runnable without any API key or network access.
- **Accessibility** — accessibility needs are treated as a first-class routing input
  (not an afterthought), the UI has visible focus states, `aria-live` on the chat
  log for screen readers, and respects `prefers-reduced-motion`.
