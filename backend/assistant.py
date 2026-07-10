"""
assistant.py

Core GenAI logic for the Smart Stadium Fan Assistant.

Design principle: the LLM is never asked to "know" stadium facts from
memory. Every request is grounded by injecting the current stadium
knowledge base + live (simulated) crowd data into the system prompt
("RAG-lite"). This keeps answers accurate and auditable, and means the
knowledge base can be swapped for a real stadium's data without touching
the assistant logic.
"""

import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler

import anthropic
from crowd_sim import get_crowd_trends, get_live_crowd_levels, recommend_gate

DATA_PATH = Path(__file__).parent / "data" / "stadium.json"
MODEL = "claude-sonnet-4-5"
DEFAULT_MODELS = {
    "anthropic": MODEL,
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-3.5-flash",
    "openai_compatible": "gpt-4.1-mini",
}


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward user-supplied credentials to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

SYSTEM_PROMPT_TEMPLATE = """You are the official Smart Stadium Assistant for the {stadium_name}, \
supporting fans, volunteers, and venue staff during a FIFA World Cup 2026 match.

{accessibility_profile_instruction}

Your responsibilities:
- Navigation: help fans find gates, seats, restrooms, concessions, and exits.
- Accessibility: proactively surface accessible routes, wheelchair seating, \
sensory rooms, and companion seating whenever relevant.
- Crowd-aware guidance: use the live crowd data below to steer fans toward \
less congested gates/routes, especially near kickoff.
- Transportation & sustainability: recommend shuttle/rail/rideshare options, \
favoring shared/public transport when it fits the fan's timing.
- Multilingual support: always reply in the same language the user wrote in, \
even if the knowledge base below is in English.

Hard rules:
- Only state facts that are present in the STADIUM KNOWLEDGE BASE or LIVE \
CROWD DATA below. Never invent gate numbers, wait times, or services.
- If asked something the data doesn't cover, say so plainly and suggest \
contacting Guest Services rather than guessing.
- Keep answers short, concrete, and actionable (fans are often mid-walk, \
in a crowd, or in a hurry). Use step-like clarity, not long paragraphs.
- If the user's request suggests an accessibility need, mention the \
relevant accessibility service even if they didn't ask for it directly.

STADIUM KNOWLEDGE BASE:
{knowledge_base}

LIVE CROWD DATA (0=empty, 1=at capacity, refreshed each request):
{crowd_data}

TRANSPORT STATUS:
{transport_status}

RECOMMENDED GATE RIGHT NOW (least congested): {recommended_gate}
RECOMMENDED ACCESSIBLE GATE RIGHT NOW: {recommended_accessible_gate}

LANGUAGE CONFIGURATION: {lang_instruction}
"""


def _load_stadium_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(
    closed_gates: set[str] | None = None,
    transport_delay: bool = False,
    surged_gates: dict[str, float] | None = None,
    volunteers_active: dict[str, bool] | None = None,
    accessibility_profile: str | None = None,
    language: str = "en",
) -> str:
    data = _load_stadium_data()
    crowd = get_live_crowd_levels(
        data["crowd_baseline"],
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )
    best_gate = recommend_gate(
        data["crowd_baseline"],
        excluded_gates=closed_gates,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )
    best_accessible_gate = recommend_gate(
        data["crowd_baseline"],
        accessible_only=True,
        gates_meta=data["gates"],
        excluded_gates=closed_gates,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )

    crowd_prompt = {}
    for gate_name, info in crowd.items():
        if closed_gates and gate_name in closed_gates:
            crowd_prompt[gate_name] = {"score": 0.0, "label": "CLOSED"}
        else:
            crowd_prompt[gate_name] = info

    transport_status = (
        "WARNING: There is an active transport delay incident. Roads and shuttle routes "
        "are heavily delayed. Advise fans to allow extra transit time and use rail fallback options."
        if transport_delay else "Normal transit operations."
    )

    accessibility_profile_instruction = ""
    if accessibility_profile:
        profile_names = {
            "wheelchair": "Wheelchair / step-free route",
            "sensory": "Sensory-friendly route",
            "companion": "Companion seating assistance",
            "asl": "ASL interpretation support",
            "walking": "Reduced walking distance"
        }
        name = profile_names.get(accessibility_profile, accessibility_profile)
        accessibility_profile_instruction = (
            f"ACTIVE USER ACCESSIBILITY PROFILE: {name}. "
            f"Adjust routing instructions to strictly adhere to this profile. "
            f"Proactively mention relevant services (e.g. wheelchair rentals at Gate D, "
            f"ASL interpretation requests via Guest Services, sensory kits at Gate D, "
            f"sensory room in Zone Z4, or shuttle drop-offs close to gates)."
        )

    lang_directives = {
        "en": "Respond strictly in English.",
        "es": "Responde estrictamente en Español (Spanish). Traduce todos los nombres de servicios, estados de puertas y tiempos de espera al Español.",
        "fr": "Répondez strictement en Français (French). Traduisez tous les noms de services, statuts des portes et temps d'attente en Français."
    }
    lang_instruction = lang_directives.get(language, lang_directives['en'])

    return SYSTEM_PROMPT_TEMPLATE.format(
        stadium_name=data["stadium_name"],
        knowledge_base=json.dumps(data, indent=2),
        crowd_data=json.dumps(crowd_prompt, indent=2),
        transport_status=transport_status,
        recommended_gate=best_gate or "none (escalate)",
        recommended_accessible_gate=best_accessible_gate or "none (escalate)",
        accessibility_profile_instruction=accessibility_profile_instruction,
        lang_instruction=lang_instruction,
    )


def translate_demo_text(text: str, lang: str) -> str:
    if lang == "en":
        return text
    translations_es = {
        "is currently CLOSED. Please route to the": "está CERRADA actualmente. Por favor diríjase a la",
        "instead.": "en su lugar.",
        "is available in your zone,": "está disponible en su zona,",
        "reached via Gate": "a la que se accede por la Puerta",
        "is available at": "está disponible en",
        "Tell me your section for the closest option.": "Dígame su sección para la opción más cercana.",
        "location could not be determined.": "ubicación no se pudo determinar.",
        "Section": "Sección",
        "is in": "está en la",
        "Use Gate": "Use la Puerta",
        "amenities there include": "las comodidades allí incluyen",
        "is unavailable.": "no está disponible.",
        "is CLOSED due to an operational incident.": "está CERRADA debido a un incidente operativo.",
        "is currently": "está actualmente",
        "and has step-free access": "y tiene acceso sin escalones",
        "it is the least-congested gate right now": "es la puerta menos congestionada en este momento",
        "Wheelchair Profile Active: Recommending step-free concourses. Enter via": "Perfil de silla de ruedas activo: recomendación de pasillos sin escalones. Ingrese por la",
        "Wheelchair rental is at Gate D Guest Services.": "El alquiler de sillas de ruedas está en el Servicio de Atención al Invitado de la Puerta D.",
        "Sensory-Friendly Profile Active: Recommending low-congestion entry via": "Perfil sensorial activo: recomendación de entrada de baja congestión por la",
        "Sensory kits are available at Gate D Guest Services. Quiet sensory room is in Zone Z4.": "Los kits sensoriales están disponibles en la Puerta D. La sala sensorial silenciosa está en la Zona Z4.",
        "Companion Seating Profile Active: Companion seating requests are managed at Gate A and D supervisor desks.": "Perfil de asientos acompañantes activo: las solicitudes se gestionan en las mesas de supervisores de las Puertas A y D.",
        "ASL Profile Active: Request ASL interpretation 48 hours in advance at Gate D Guest Services.": "Perfil ASL activo: solicite interpretación de ASL con 48 horas de anticipación en Guest Services de la Puerta D.",
        "Short Distance Profile Active: Avoid Meadowlands Rail walk. Recommending shuttle loops or Lot B. Entrance:": "Perfil de corta distancia activo: evite la caminata a la estación. Se recomiendan lanzaderas o el Lote B. Entrada:",
        "quiet sensory room": "sala sensorial silenciosa",
        "nursing room": "sala de lactancia",
        "first aid": "primeros auxilios",
        "restrooms": "baños",
        "restroom": "baño",
        "toilet": "inodoro",
        "bathroom": "baño",
        "Demo mode (no API key):": "Modo de demostración (sin clave API):",
        "transport delay is active": "el retraso de transporte está activo",
        "Normal transit operations.": "Operaciones normales de tránsito.",
        "Accessible parking is Lot B.": "El estacionamiento accesible es el Lote B."
    }
    
    translations_fr = {
        "is currently CLOSED. Please route to the": "est actuellement FERMÉE. Veuillez vous diriger vers la",
        "instead.": "à la place.",
        "is available in your zone,": "est disponible dans votre zone,",
        "reached via Gate": "accessible par la Porte",
        "is available at": "est disponible à",
        "Tell me your section for the closest option.": "Dites-moi votre section pour l'option la plus proche.",
        "location could not be determined.": "l'emplacement n'a pas pu être déterminé.",
        "Section": "Section",
        "is in": "est dans la",
        "Use Gate": "Utilisez la Porte",
        "amenities there include": "les services sur place comprennent",
        "is unavailable.": "est indisponible.",
        "is CLOSED due to an operational incident.": "est FERMÉE en raison d'un incident opérationnel.",
        "is currently": "est actuellement",
        "and has step-free access": "et dispose d'un accès sans marche",
        "it is the least-congested gate right now": "c'est la porte la moins encombrée actuellement",
        "Wheelchair Profile Active: Recommending step-free concourses. Enter via": "Profil fauteuil roulant actif: recommandation de halls sans marche. Entrez par la",
        "Wheelchair rental is at Gate D Guest Services.": "La location de fauteuils roulants s'effectue au comptoir d'accueil de la Porte D.",
        "Sensory-Friendly Profile Active: Recommending low-congestion entry via": "Profil sensoriel actif: recommandation d'une entrée à faible affluence via la",
        "Sensory kits are available at Gate D Guest Services. Quiet sensory room is in Zone Z4.": "Les kits sensoriels sont disponibles à la Porte D. La salle sensorielle calme est en Zone Z4.",
        "Companion Seating Profile Active: Companion seating requests are managed at Gate A and D supervisor desks.": "Profil sièges compagnon actif: les demandes de sièges compagnons sont gérées aux bureaux des superviseurs des Portes A et D.",
        "ASL Profile Active: Request ASL interpretation 48 hours in advance at Gate D Guest Services.": "Profil ASL actif: demandez une interprétation ASL 48 heures à l'avance à la Porte D.",
        "Short Distance Profile Active: Avoid Meadowlands Rail walk. Recommending shuttle loops or Lot B. Entrance:": "Profil distance réduite actif: évitez la marche vers la gare. Navettes ou Lot B recommandés. Entrée:",
        "quiet sensory room": "salle sensorielle calme",
        "nursing room": "salle d'allaitement",
        "first aid": "premiers secours",
        "restrooms": "toilettes",
        "restroom": "toilette",
        "toilet": "toilette",
        "bathroom": "salle de bain",
        "Demo mode (no API key):": "Mode démo (sans clé API) :",
        "transport delay is active": "le retard de transport est actif",
        "Normal transit operations.": "Opérations de transit normales.",
        "Accessible parking is Lot B.": "Le parking accessible est le Lot B."
    }
    translations = translations_es if lang == "es" else translations_fr
    replaced_text = text
    for en_phrase, target_phrase in translations.items():
        replaced_text = replaced_text.replace(en_phrase, target_phrase)
    return replaced_text


class StadiumAssistant:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Keeping the client optional lets the full UI run locally without a
        # paid API key.  In that case ask() provides a small, grounded demo
        # response using the same stadium data and crowd simulation.
        self.client = anthropic.Anthropic(api_key=key) if key else None

    @staticmethod
    def _demo_reply(
        user_message: str,
        closed_gates: set[str] | None = None,
        transport_delay: bool = False,
        surged_gates: dict[str, float] | None = None,
        volunteers_active: dict[str, bool] | None = None,
        accessibility_profile: str | None = None,
    ) -> str:
        """Return practical, deterministic guidance from the local venue data."""
        data = _load_stadium_data()
        status = get_live_status(
            closed_gates=closed_gates,
            transport_delay=transport_delay,
            surged_gates=surged_gates,
            volunteers_active=volunteers_active,
        )
        message = user_message.lower()
        accessible_terms = (
            "accessible", "accessibility", "wheelchair", "step-free", "mobility",
            "sensory", "asl", "companion",
        )

        def zone_for_section() -> dict | None:
            match = re.search(r"\b([1-3]\d{2})\b", message)
            if not match:
                return None
            section = int(match.group(1))
            for zone in data["zones"]:
                bounds = re.search(r"Sections (\d+)-(\d+)", zone["name"])
                if bounds and int(bounds.group(1)) <= section <= int(bounds.group(2)):
                    return zone
            return None

        # Build profile note
        profile_note = ""
        if accessibility_profile == "wheelchair":
            best_acc = status["recommended_accessible_gate"]
            profile_note = (
                f"Wheelchair Profile Active: Recommending step-free concourses. "
                f"Enter via {best_acc or 'accessible gate'}. Wheelchair rental is at Gate D Guest Services."
            )
        elif accessibility_profile == "sensory":
            best_g = status["recommended_gate"]
            profile_note = (
                f"Sensory-Friendly Profile Active: Recommending low-congestion entry via {best_g or 'gate'}. "
                f"Sensory kits are available at Gate D Guest Services. Quiet sensory room is in Zone Z4."
            )
        elif accessibility_profile == "companion":
            profile_note = (
                f"Companion Seating Profile Active: Companion seating requests are managed at Gate A and D supervisor desks."
            )
        elif accessibility_profile == "asl":
            profile_note = (
                f"ASL Profile Active: Request ASL interpretation 48 hours in advance at Gate D Guest Services."
            )
        elif accessibility_profile == "walking":
            best_g = status["recommended_gate"]
            profile_note = (
                f"Short Distance Profile Active: Avoid Meadowlands Rail walk. Recommending shuttle loops or Lot B. Entrance: {best_g or 'gate'}."
            )

        def gate_reason(gate_name: str) -> str:
            live_gate = next((gate for gate in status["gates"] if gate["name"] == gate_name), None)
            if not live_gate:
                return f"{gate_name} is unavailable."
            if live_gate["closed"]:
                return f"{gate_name} is CLOSED due to an operational incident."
            reason = f"{gate_name} is currently {live_gate['label']} ({round(live_gate['score'] * 100)}%)"
            if live_gate["accessible"]:
                reason += " and has step-free access"
            if gate_name == status["recommended_gate"]:
                reason += "; it is the least-congested gate right now"
            return reason + "."

        # Check if they ask about a gate that is closed
        closed_match = None
        for gate in status["gates"]:
            if gate["closed"] and gate["name"].lower() in message:
                closed_match = gate["name"]
                break

        if closed_match:
            rec_str = f"recommended gate {status['recommended_gate']}" if status["recommended_gate"] else "venue supervisor"
            reply = (
                f"Demo mode (no API key): {closed_match} is currently CLOSED. "
                f"Please route to the {rec_str} instead."
            )
        else:
            zone = zone_for_section()
            amenity_queries = {
                "quiet_sensory_room": ("sensory", "quiet room", "quiet"),
                "nursing_room": ("nursing", "baby", "infant"),
                "first_aid": ("first aid", "medical", "medic"),
                "restrooms": ("restroom", "restrooms", "toilet", "bathroom"),
            }
            requested_amenity = next(
                (amenity for amenity, terms in amenity_queries.items() if any(term in message for term in terms)),
                None,
            )

            if requested_amenity:
                if zone and requested_amenity in zone["amenities"]:
                    reply = (
                        "Demo mode (no API key): "
                        f"{requested_amenity.replace('_', ' ')} is available in your zone, "
                        f"{zone['name']}, reached via Gate {zone['nearest_gate']}."
                    )
                else:
                    locations = [z for z in data["zones"] if requested_amenity in z["amenities"]]
                    if locations:
                        places = "; ".join(
                            f"{item['name']} (Gate {item['nearest_gate']})" for item in locations
                        )
                        reply = (
                            "Demo mode (no API key): "
                            f"{requested_amenity.replace('_', ' ')} is available at {places}. "
                            "Tell me your section for the closest option."
                        )
                    else:
                        reply = f"Demo mode (no API key): {requested_amenity.replace('_', ' ')} location could not be determined."
            elif zone:
                amenities = ", ".join(item.replace("_", " ") for item in zone["amenities"])
                gate_name = f"Gate {zone['nearest_gate']}"
                reply = (
                    "Demo mode (no API key): "
                    f"Section {re.search(r'\b([1-3]\d{2})\b', message).group(1)} is in {zone['name']}. "
                    f"Use {gate_name}; amenities there include {amenities}. {gate_reason(gate_name)}"
                )
            elif any(term in message for term in ("airport", "downtown", "train", "rail", "shuttle", "transport", "home", "rideshare", "parking", "leaving")):
                transport = data["transport"]
                access_note = (
                    f" Accessible parking is {next(lot for lot in transport['parking_lots'] if 'Accessible' in lot)}."
                    if any(term in message for term in accessible_terms) else ""
                )
                prefix = "Demo mode (no API key): "
                if status["transport_delay"]:
                    prefix += "⚠️ NOTICE: Active transport delay in progress. Expect major shuttle delays; rail is recommended. "
                if accessibility_profile == "walking":
                    prefix += "🚶 Short Distance Advice: Avoid Meadowlands Rail walk (12 min); use shuttles or Lot B. "
                
                if "airport" in message:
                    line = next(line for line in transport["shuttle_lines"] if "Airport" in line["name"])
                    reply = (
                        f"{prefix}take the "
                        f"{line['name']}, which runs every {line['frequency_minutes']} minutes and has its "
                        f"last departure {line['last_departure_after_match_minutes']} minutes after the match.{access_note}"
                    )
                elif "downtown" in message:
                    line = next(line for line in transport["shuttle_lines"] if "Downtown" in line["name"])
                    reply = (
                        f"{prefix}take the "
                        f"{line['name']}, which runs every {line['frequency_minutes']} minutes and has its "
                        f"last departure {line['last_departure_after_match_minutes']} minutes after the match.{access_note}"
                    )
                elif "rideshare" in message:
                    reply = f"{prefix}rideshare pickup is at {transport['rideshare_pickup_zone']}.{access_note}"
                else:
                    reply = (
                        f"{prefix}for public transport, "
                        f"{transport['rail_station']} is a {transport['rail_walk_minutes']}-minute walk. "
                        f"For rideshare, use {transport['rideshare_pickup_zone']}.{access_note}"
                    )
            elif any(term in message for term in accessible_terms):
                services = data["accessibility_services"]
                gate = status["recommended_accessible_gate"]
                if not gate:
                    reply = (
                        "Demo mode (no API key): No accessible gates are currently open. "
                        "Please contact Guest Services immediately."
                    )
                else:
                    reply = (
                        "Demo mode (no API key): use "
                        f"{gate} for step-free entry. {gate_reason(gate)} "
                        f"Wheelchair rental and sensory kits: {services['wheelchair_rental_location']}. "
                        f"Companion seating: {services['companion_seating']}"
                    )
            elif any(term in message for term in ("gate", "entrance", "line", "queue", "fastest", "shortest")):
                gate = status["recommended_gate"]
                if accessibility_profile == "wheelchair":
                    gate = status["recommended_accessible_gate"]
                if not gate:
                    reply = (
                        "Demo mode (no API key): All gates are currently closed. "
                        "Please contact Guest Services."
                    )
                else:
                    reply = f"Demo mode (no API key): use {gate}. {gate_reason(gate)}"
            else:
                best_g = status["recommended_gate"]
                best_acc = status["recommended_accessible_gate"]
                if accessibility_profile == "wheelchair":
                    best_g = best_acc

                gate_part = f"the least-congested gate right now is {best_g}. {gate_reason(best_g)}" if best_g else "all gates are closed."
                acc_part = f"For step-free entry, use {best_acc}." if best_acc else "No accessible gates are open."
                reply = (
                    "Demo mode (no API key): "
                    f"{gate_part} {acc_part} "
                    "Ask about a section, amenity, accessibility need, or journey home."
                )

        # Append the profile note if active and not already included
        if profile_note and profile_note not in reply:
            reply += f" [Active Profile: {profile_note}]"

        return reply

    @staticmethod
    def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
        """Make a JSON request without following redirects (keys must not be forwarded)."""
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        opener = build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("The selected provider request failed.") from exc

    def _ask_openai_compatible(
        self, user_message: str, history: list[dict], api_key: str, model: str, base_url: str,
        closed_gates: set[str] | None = None,
        transport_delay: bool = False,
        surged_gates: dict[str, float] | None = None,
        volunteers_active: dict[str, bool] | None = None,
        accessibility_profile: str | None = None,
        language: str = "en",
    ) -> str:
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("Custom endpoint must start with http:// or https://")
        endpoint = f"{base_url}/v1/chat/completions"
        messages = [{"role": "system", "content": build_system_prompt(closed_gates, transport_delay, surged_gates, volunteers_active, accessibility_profile, language=language)}] + history + [
            {"role": "user", "content": user_message}
        ]
        response = self._post_json(
            endpoint,
            {"model": model, "messages": messages, "max_tokens": 600},
            {"Authorization": f"Bearer {api_key}"},
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The selected provider returned an unexpected response.") from exc

    def _ask_gemini(
        self, user_message: str, history: list[dict], api_key: str, model: str,
        closed_gates: set[str] | None = None,
        transport_delay: bool = False,
        surged_gates: dict[str, float] | None = None,
        volunteers_active: dict[str, bool] | None = None,
        accessibility_profile: str | None = None,
        language: str = "en",
    ) -> str:
        contents = [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in history + [{"role": "user", "content": user_message}]
        ]
        response = self._post_json(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:generateContent",
            {
                "system_instruction": {"parts": [{"text": build_system_prompt(closed_gates, transport_delay, surged_gates, volunteers_active, accessibility_profile, language=language)}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": 600},
            },
            {"x-goog-api-key": api_key},
        )
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The selected provider returned an unexpected response.") from exc

    def ask(
        self,
        user_message: str,
        history: list[dict] | None = None,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        closed_gates: set[str] | None = None,
        transport_delay: bool = False,
        surged_gates: dict[str, float] | None = None,
        volunteers_active: dict[str, bool] | None = None,
        accessibility_profile: str | None = None,
        language: str = "en",
    ) -> str:
        """
        history: list of {"role": "user"|"assistant", "content": str}
        """
        history = history or []
        if provider == "demo":
            reply = self._demo_reply(
                user_message,
                closed_gates=closed_gates,
                transport_delay=transport_delay,
                surged_gates=surged_gates,
                volunteers_active=volunteers_active,
                accessibility_profile=accessibility_profile,
            )
            return translate_demo_text(reply, language)

        model = model or DEFAULT_MODELS[provider]
        if provider == "anthropic":
            client = self.client if not api_key else anthropic.Anthropic(api_key=api_key)
            if client is None:
                raise ValueError("An Anthropic API key is required for this provider.")
            response = client.messages.create(
                model=model,
                max_tokens=600,
                system=build_system_prompt(closed_gates, transport_delay, surged_gates, volunteers_active, accessibility_profile, language=language),
                messages=history + [{"role": "user", "content": user_message}],
            )
            return "".join(block.text for block in response.content if block.type == "text")

        if not api_key:
            raise ValueError("An API key is required for the selected provider.")
        if provider == "gemini":
            return self._ask_gemini(user_message, history, api_key, model, closed_gates, transport_delay, surged_gates, volunteers_active, accessibility_profile, language=language)
        if provider in ("openai", "openai_compatible"):
            endpoint = "https://api.openai.com" if provider == "openai" else (base_url or "")
            return self._ask_openai_compatible(user_message, history, api_key, model, endpoint, closed_gates, transport_delay, surged_gates, volunteers_active, accessibility_profile, language=language)
        raise ValueError("Unsupported provider.")


def get_live_status(
    closed_gates: set[str] | None = None,
    transport_delay: bool = False,
    surged_gates: dict[str, float] | None = None,
    volunteers_active: dict[str, bool] | None = None,
    alert_threshold: float = 0.75,
) -> dict:
    """Exposed for a small dashboard / health endpoint - live crowd + recommended gate,
    independent of the LLM, useful for staff-facing views that need raw numbers."""
    data = _load_stadium_data()
    closed_gates = closed_gates or set()
    surged_gates = surged_gates or {}
    volunteers_active = volunteers_active or {}
    
    crowd = get_live_crowd_levels(
        data["crowd_baseline"],
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )
    trends = get_crowd_trends(
        data["crowd_baseline"],
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )

    gates_by_id = {g["id"]: g for g in data["gates"]}
    gates = []
    for gate_name, info in crowd.items():
        gate_id = gate_name.split()[-1]
        meta = gates_by_id.get(gate_id, {})
        gates.append({
            "name": gate_name,
            "score": info["score"],
            "label": info["label"],
            "accessible": meta.get("accessible", False),
            "typical_wait_minutes": meta.get("typical_wait_minutes"),
            "closed": gate_name in closed_gates,
            "trend_30_minutes": trends[gate_name],
            "trend_delta": round((trends[gate_name][-1] - trends[gate_name][0]) * 100),
            "surged": gate_name in surged_gates,
            "volunteers_active": volunteers_active.get(gate_name, False),
        })

    recommended_gate = recommend_gate(
        data["crowd_baseline"],
        excluded_gates=closed_gates,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )
    recommended_accessible_gate = recommend_gate(
        data["crowd_baseline"],
        accessible_only=True,
        gates_meta=data["gates"],
        excluded_gates=closed_gates,
        surged_gates=surged_gates,
        volunteers_active=volunteers_active,
    )

    alerts = []
    for g in gates:
        if g["closed"]:
            continue
        if g["score"] >= alert_threshold:
            if g["volunteers_active"]:
                alerts.append(f"Gate Surge Alert: {g['name']} is at {round(g['score']*100)}% (Mitigated: Volunteers deployed).")
            else:
                suggested = recommended_gate or "another open gate"
                alerts.append(
                    f"Gate Surge Alert: {g['name']} is at {round(g['score']*100)}%. "
                    f"Suggested rerouting: Move volunteers from {g['name']} to {suggested}."
                )

    for gate in sorted(closed_gates):
        alerts.append(f"Incident: {gate} is closed — remove it from fan and volunteer routing.")
        
    if transport_delay:
        alerts.append("Incident: Transport delay mode is active — advise fans to allow extra time and use the rail fallback.")

    if recommended_gate:
        recommended_info = next(gate for gate in gates if gate["name"] == recommended_gate)
        recommendation_reason = (
            f"{recommended_gate} is the lowest-congestion open gate at "
            f"{round(recommended_info['score'] * 100)}%."
        )
    else:
        recommendation_reason = "No open gate is available; escalate to venue command."

    return {
        "stadium_name": data["stadium_name"],
        "crowd": crowd,
        "gates": gates,
        "alerts": alerts,
        "recommended_gate": recommended_gate,
        "recommended_accessible_gate": recommended_accessible_gate,
        "recommendation_reason": recommendation_reason,
        "transport_delay": transport_delay,
        "alert_threshold": alert_threshold,
    }
