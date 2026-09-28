"""
llm_service.py — DealMind AI/LLM Intelligence Component
=========================================================
Responsible: Manasa
Project: DealMind — Sales Conversation Intelligence Agent

Public API (for Divya's backend to import):
    from llm_service import analyze_conversation, generate_meeting_brief

MOCK_MODE = True  → Works without any API key (development / demo).
MOCK_MODE = False → Uses real Hugging Face LLM via Responses API.
                    Requires HF_TOKEN to be set in .env.
"""

import os
import json
import re

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load HF_TOKEN (and GROQ_API_KEY if present) from .env

MOCK_MODE = False  # Set to True to use keyword-based mock (no API key needed)

# Hugging Face router — OpenAI-compatible endpoint
_HF_BASE_URL = "https://router.huggingface.co/v1"
_HF_MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Empty result templates
# ---------------------------------------------------------------------------

def _empty_analysis() -> dict:
    """Return a blank analysis result matching the DealMind schema."""
    return {
        "customer_name": "",
        "company": "",
        "objections": [],
        "competitors": [],
        "requirements": [],
        "pricing_discussions": [],
        "promises": [],
        "follow_ups": [],
        "sentiment": "",
        "summary": "",
    }


def _empty_brief() -> dict:
    """Return a blank meeting brief matching the DealMind schema."""
    return {
        "customer_summary": "",
        "key_objections": [],
        "competitors": [],
        "requirements": [],
        "open_commitments": [],
        "suggested_talking_points": [],
    }


# ---------------------------------------------------------------------------
# Mock implementations  (always available as fallback)
# ---------------------------------------------------------------------------

# Simple keyword lists for mock extraction — deterministic, no NLP required.
_COMPETITOR_KEYWORDS = [
    "salesforce", "hubspot", "zoho", "pipedrive", "microsoft dynamics",
    "oracle", "sap", "freshsales", "monday", "copper",
]

_NEGATIVE_SENTIMENT_WORDS = [
    "expensive", "too high", "costly", "overpriced", "concern", "worried",
    "problem", "issue", "difficult", "complicated", "unhappy", "disappointed",
    "reject", "refuse", "not interested",
]

_POSITIVE_SENTIMENT_WORDS = [
    "like", "likes", "love", "loves", "great", "excellent", "good",
    "impressed", "happy", "interested", "keen", "positive", "appreciate",
    "satisfied", "perfect",
]

_PROMISE_KEYWORDS = [
    "promised", "will send", "will share", "will provide", "will follow",
    "will schedule", "committed", "going to send", "going to share",
]

_FOLLOWUP_KEYWORDS = [
    "send", "share", "schedule", "follow up", "follow-up", "provide",
    "arrange", "call", "email", "demo", "meeting",
]


def _extract_mock_sentiment(text: str) -> str:
    """
    Derive a simple sentiment label from keyword counts.
    Returns: 'positive', 'negative', 'mixed', or 'neutral'.
    """
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE_SENTIMENT_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_SENTIMENT_WORDS if w in lower)

    if pos > 0 and neg > 0:
        return "mixed"
    if pos > 0:
        return "positive"
    if neg > 0:
        return "negative"
    return "neutral"


def _analyze_mock(conversation: str) -> dict:
    """
    Deterministic mock analysis for development / demo / testing.

    Uses keyword matching and light regex — no API key required.
    Produces predictable structured output for the standard DealMind
    test conversation about Rahul from ABC Technologies.
    """
    result = _empty_analysis()
    lower = conversation.lower()

    # Protect abbreviations like "Rs." so the sentence splitter doesn't break them
    protected = re.sub(r"\b(Rs|Mr|Ms|Mrs|Dr|Prof|No|St|vs)\.", r"\1<DOT>", conversation)
    sentences = [s.strip() for s in re.split(r"[.!?]", protected) if s.strip()]
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    # --- customer_name & company ---
    name_match = re.search(
        r"\b([A-Z][a-z]+)\s+from\s+([A-Z][A-Za-z\s&]+?)(?:\s+(?:likes|considers|thinks|needs|wants|is|has|will))",
        conversation,
    )
    if name_match:
        result["customer_name"] = name_match.group(1)
        result["company"] = name_match.group(2).strip()
    else:
        words = conversation.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0 and len(word) > 2:
                result["customer_name"] = word.strip(".,")
                break

    # --- objections ---
    objection_patterns = [
        r"thinks?\s+(?:the\s+)?(?:our\s+)?(?:price|cost|pricing)\s+is\s+(?:too\s+)?\w+",
        r"considers?\s+.+?(?:too\s+)?\w+\s+(?:expensive|costly|high)",
        r"price\s+is\s+(?:too\s+)?\w+",
        r"(?:too\s+expensive|too\s+high|overpriced)",
        r"(?:concerned?|worried?)\s+about\s+\w+",
    ]
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(re.search(p, s_lower) for p in objection_patterns):
            clean = sentence.strip()
            if clean and clean not in result["objections"]:
                result["objections"].append(clean)

    if not result["objections"]:
        for sentence in sentences:
            if any(w in sentence.lower() for w in _NEGATIVE_SENTIMENT_WORDS):
                result["objections"].append(sentence.strip())

    # --- competitors ---
    for competitor in _COMPETITOR_KEYWORDS:
        if competitor in lower:
            result["competitors"].append(competitor.title())

    # --- requirements ---
    req_patterns = [
        r"needs?\s+([a-zA-Z][a-zA-Z\s,]+?)(?:\.|\Z)",
        r"requires?\s+([a-zA-Z][a-zA-Z\s,]+?)(?:\.|\Z)",
        r"looking\s+for\s+([a-zA-Z][a-zA-Z\s,]+?)(?:\.|\Z)",
        r"wants?\s+([a-zA-Z][a-zA-Z\s,]+?)(?:\.|\Z)",
    ]
    for pattern in req_patterns:
        for match in re.finditer(pattern, lower):
            raw_req = match.group(1).strip(" .,")
            parts = re.split(r"\s+and\s+|,\s*", raw_req)
            for part in parts:
                req = part.strip(" .,")
                if req and len(req) > 3:
                    req_cap = req.capitalize()
                    if req_cap not in result["requirements"]:
                        result["requirements"].append(req_cap)

    # --- pricing_discussions ---
    price_patterns = [
        r"(?:rs\.?\s*[\d,]+(?:\s*(?:thousand|lakh|crore|k|l))?)",
        r"\$\s*[\d,]+(?:\s*(?:thousand|k))?",
        r"[\d,]+\s*(?:rs|rupees|inr)",
    ]
    for pattern in price_patterns:
        for match in re.finditer(pattern, lower):
            for sentence in sentences:
                if match.group() in sentence.lower():
                    entry = sentence.strip()
                    if entry not in result["pricing_discussions"]:
                        result["pricing_discussions"].append(entry)

    if not result["pricing_discussions"]:
        for sentence in sentences:
            if any(w in sentence.lower() for w in ["price", "cost", "budget", "pricing", "expensive"]):
                result["pricing_discussions"].append(sentence)

    # --- promises ---
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in _PROMISE_KEYWORDS):
            result["promises"].append(sentence.strip())

    # --- follow_ups ---
    for promise in result["promises"]:
        action = re.sub(r"^I\s+", "", promise, flags=re.IGNORECASE).strip()
        if action and action not in result["follow_ups"]:
            result["follow_ups"].append(action)

    for sentence in sentences:
        if any(kw in sentence.lower() for kw in _FOLLOWUP_KEYWORDS) and sentence not in result["promises"]:
            action = sentence.strip()
            if action not in result["follow_ups"]:
                result["follow_ups"].append(action)

    result["follow_ups"] = list(dict.fromkeys(result["follow_ups"]))

    # --- sentiment ---
    result["sentiment"] = _extract_mock_sentiment(conversation)

    # --- summary ---
    customer = result["customer_name"] or "The customer"
    company_str = f" from {result['company']}" if result["company"] else ""
    competitors_str = (
        f" They are also evaluating {', '.join(result['competitors'])}."
        if result["competitors"] else ""
    )
    reqs_str = (
        f" Key requirements include: {', '.join(result['requirements'][:3])}."
        if result["requirements"] else ""
    )
    objections_str = (
        f" Main objection: {result['objections'][0]}."
        if result["objections"] else ""
    )
    promises_str = (
        f" Salesperson promised: {result['promises'][0]}."
        if result["promises"] else ""
    )
    result["summary"] = (
        f"{customer}{company_str} showed {result['sentiment']} interest in our product."
        f"{objections_str}{competitors_str}{reqs_str}{promises_str}"
    ).strip()

    return result


def _generate_brief_mock(memories: list) -> dict:
    """
    Deterministic mock meeting-brief generator for development / demo / testing.

    Builds a pre-meeting briefing from memory strings using keyword scanning —
    no API key required.
    """
    result = _empty_brief()

    all_text = " ".join(memories).lower()
    full_text = " ".join(memories)

    # --- customer_summary ---
    name_match = re.search(
        r"\b([A-Z][a-z]+)\s+from\s+([A-Z][A-Za-z\s&]+?)(?:\s+(?:considers?|is|needs?|has|will|wants?))",
        full_text,
    )
    customer_name = name_match.group(1) if name_match else "The customer"
    company = name_match.group(2).strip() if name_match else ""

    result["customer_summary"] = (
        f"{customer_name}{' from ' + company if company else ''} is an active prospect. "
        f"Multiple interactions have been recorded covering pricing, requirements, and competitive evaluation."
    )

    # --- key_objections ---
    for memory in memories:
        if any(w in memory.lower() for w in _NEGATIVE_SENTIMENT_WORDS):
            objection = memory.strip().rstrip(".")
            if objection not in result["key_objections"]:
                result["key_objections"].append(objection)

    # --- competitors ---
    for competitor in _COMPETITOR_KEYWORDS:
        if competitor in all_text:
            cap = competitor.title()
            if cap not in result["competitors"]:
                result["competitors"].append(cap)

    # --- requirements ---
    req_patterns = [
        r"needs?\s+([a-zA-Z\s]+?)(?:\.|,|$)",
        r"requires?\s+([a-zA-Z\s]+?)(?:\.|,|$)",
    ]
    for memory in memories:
        for pattern in req_patterns:
            for match in re.finditer(pattern, memory.lower()):
                req = match.group(1).strip(" .,")
                if req and len(req) > 3:
                    req_cap = req.capitalize()
                    if req_cap not in result["requirements"]:
                        result["requirements"].append(req_cap)

    # --- open_commitments ---
    commitment_keywords = [
        "promised", "will send", "will provide", "will share",
        "committed", "going to", "agreed to",
    ]
    for memory in memories:
        if any(kw in memory.lower() for kw in commitment_keywords):
            commitment = memory.strip().rstrip(".")
            if commitment not in result["open_commitments"]:
                result["open_commitments"].append(commitment)

    # --- suggested_talking_points ---
    talking_points = []
    for objection in result["key_objections"]:
        talking_points.append(f"Address pricing concern: {objection.rstrip('.')}")
    for competitor in result["competitors"]:
        talking_points.append(f"Prepare differentiation talking points against {competitor}")
    for req in result["requirements"]:
        talking_points.append(f"Demonstrate capability: {req}")
    for commitment in result["open_commitments"]:
        talking_points.append(f"Follow up on commitment: {commitment.rstrip('.')}")
    if any("expensive" in m.lower() or "too high" in m.lower() or "pricing" in m.lower() for m in memories):
        talking_points.append("Present ROI and value justification relative to the quoted price")

    result["suggested_talking_points"] = talking_points
    return result


# ---------------------------------------------------------------------------
# Hugging Face implementation  (active when MOCK_MODE = False)
# ---------------------------------------------------------------------------

def _get_hf_client():
    """
    Build and return an OpenAI client pointed at the Hugging Face router.

    Raises EnvironmentError if HF_TOKEN is not set in .env.
    """
    from openai import OpenAI  # noqa: PLC0415

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "Missing HF_TOKEN. Add your Hugging Face token to the .env file "
            "and make sure MOCK_MODE = False only after setting it."
        )
    return OpenAI(
        base_url=_HF_BASE_URL,
        api_key=token,
    )


def _parse_json_response(raw: str, template: dict) -> dict:
    """
    Safely parse a JSON string returned by the LLM.

    Handles common LLM output quirks:
    - Strips leading/trailing whitespace.
    - Strips markdown code fences (```json ... ```) if present.
    - Recovers a missing opening ``{`` brace (some models drop it).
    - Fills any missing keys from the template with empty defaults.
    - Raises ValueError with the raw text if JSON decoding still fails.
    """
    cleaned = raw.strip()

    # Remove optional markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # If the text looks like a JSON object body but is missing the opening {,
    # add it back. This covers models that strip the first character.
    if cleaned and cleaned[0] != "{" and cleaned[0] != "[":
        # Heuristic: if it starts with a quote followed by a colon pattern
        # that looks like a JSON key-value pair, wrap it
        if re.match(r'^"?\w', cleaned) and (":" in cleaned):
            cleaned = "{" + cleaned

    # Ensure it ends with } if we added an opening brace and it's missing
    if cleaned.startswith("{") and not cleaned.endswith("}"):
        cleaned = cleaned + "}"

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The LLM returned a response that could not be parsed as JSON.\n"
            f"Raw output:\n{raw}"
        ) from exc

    # Ensure every expected key is present
    for key, default in template.items():
        if key not in parsed:
            parsed[key] = default

    return parsed


def _analyze_with_huggingface(conversation: str) -> dict:
    """
    Analyze a sales conversation using the Hugging Face LLM via Responses API.

    Sends the conversation to the model with a strict system prompt requesting
    structured JSON output matching the DealMind analysis schema.
    Returns a Python dictionary guaranteed to contain all expected fields.
    """
    client = _get_hf_client()

    instructions = (
        "You are a sales intelligence AI for DealMind.\n\n"
        "Analyze the sales conversation provided by the user and return a JSON object "
        "with EXACTLY these fields:\n\n"
        "  customer_name       (string)  — Full name of the customer/prospect. "
        'Use "" if unknown.\n'
        "  company             (string)  — Customer's company name. "
        'Use "" if unknown.\n'
        "  objections          (array)   — List of objections or concerns preventing the purchase. "
        "Use [] if none.\n"
        "  competitors         (array)   — Competing products or vendors being evaluated. "
        "Use [] if none.\n"
        "  requirements        (array)   — Features or capabilities the customer requested. "
        "Use [] if none.\n"
        "  pricing_discussions (array)   — Any pricing figures, budget concerns, or discount requests. "
        "Use [] if none.\n"
        "  promises            (array)   — Specific commitments made by the salesperson. "
        "Use [] if none.\n"
        "  follow_ups          (array)   — Actions that must happen after this meeting. "
        "Use [] if none.\n"
        "  sentiment           (string)  — One of: positive, neutral, negative, mixed.\n"
        "  summary             (string)  — A concise 2-3 sentence summary useful to a salesperson.\n\n"
        "RULES:\n"
        "- Return ONLY a valid JSON object. No markdown, no explanation, no extra text.\n"
        "- Never invent facts. Only extract information explicitly present in the conversation.\n"
        "- Each array item must be a concise, factual string.\n"
        "- sentiment MUST be exactly one of: positive, neutral, negative, mixed.\n"
        "- If a field has no data, use \"\" for strings and [] for arrays."
    )

    response = client.responses.create(
        model=_HF_MODEL,
        instructions=instructions,
        input=conversation,
        reasoning={"effort": "low"},
    )

    raw = response.output_text
    return _parse_json_response(raw, _empty_analysis())


def _generate_brief_with_huggingface(memories: list) -> dict:
    """
    Generate a pre-meeting briefing from customer memory strings using the
    Hugging Face LLM via Responses API.

    Returns a Python dictionary guaranteed to contain all expected fields.
    """
    client = _get_hf_client()

    memories_text = "\n".join(f"- {m}" for m in memories)

    instructions = (
        "You are a sales preparation AI for DealMind.\n\n"
        "The user will provide a list of customer interaction memories. "
        "Using ONLY those memories as your factual source, generate a pre-meeting briefing.\n\n"
        "Return a JSON object with EXACTLY these fields:\n\n"
        "  customer_summary         (string) — Brief summary of who the customer is and their history. "
        "Base it only on the memories provided.\n"
        "  key_objections           (array)  — Important objections or concerns raised by the customer. "
        "Use [] if none found.\n"
        "  competitors              (array)  — Competing products or vendors mentioned. "
        "Use [] if none.\n"
        "  requirements             (array)  — Features or capabilities the customer needs. "
        "Use [] if none.\n"
        "  open_commitments         (array)  — Unresolved promises or commitments the salesperson made. "
        "Use [] if none.\n"
        "  suggested_talking_points (array)  — Actionable talking points for the next meeting. "
        "These may include reasonable sales recommendations derived from the memories, "
        "but must not be presented as historical customer facts.\n\n"
        "RULES:\n"
        "- Return ONLY a valid JSON object. No markdown, no explanation, no extra text.\n"
        "- Do NOT invent customer facts not present in the memories.\n"
        "- Each array item must be a concise, actionable string.\n"
        "- suggested_talking_points should be practical and useful to a salesperson."
    )

    response = client.responses.create(
        model=_HF_MODEL,
        instructions=instructions,
        input=f"Customer interaction memories:\n{memories_text}",
        reasoning={"effort": "low"},
    )

    raw = response.output_text
    return _parse_json_response(raw, _empty_brief())


# ---------------------------------------------------------------------------
# Groq stubs — UNUSED (Groq replaced by Hugging Face as real-mode provider)
# These functions are preserved here for reference but are NOT called by the
# public API. They can be safely removed once Groq is fully decommissioned.
# ---------------------------------------------------------------------------

def _get_groq_client():  # noqa: F401 — unused, kept for reference
    """[UNUSED] Returns an authenticated Groq client. Replaced by HF."""
    from groq import Groq  # noqa: PLC0415

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY.")
    return Groq(api_key=api_key)


def _analyze_with_groq(conversation: str) -> dict:  # noqa: F401 — unused, kept for reference
    """[UNUSED] Groq-based conversation analysis. Replaced by _analyze_with_huggingface()."""
    raise NotImplementedError(
        "_analyze_with_groq() is no longer the active real-mode provider. "
        "Use _analyze_with_huggingface() instead, or set MOCK_MODE = True."
    )


def _generate_brief_with_groq(memories: list) -> dict:  # noqa: F401 — unused, kept for reference
    """[UNUSED] Groq-based meeting brief generation. Replaced by _generate_brief_with_huggingface()."""
    raise NotImplementedError(
        "_generate_brief_with_groq() is no longer the active real-mode provider. "
        "Use _generate_brief_with_huggingface() instead, or set MOCK_MODE = True."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_conversation(conversation: str) -> dict:
    """
    Analyze a raw sales meeting conversation and return structured intelligence.

    Parameters
    ----------
    conversation : str
        Raw notes or transcript from a sales meeting.

    Returns
    -------
    dict with keys:
        customer_name, company, objections, competitors, requirements,
        pricing_discussions, promises, follow_ups, sentiment, summary

    Example
    -------
    >>> result = analyze_conversation("Rahul from ABC Technologies thinks Rs. 50,000 is expensive.")
    >>> result["customer_name"]
    'Rahul'

    Notes
    -----
    - MOCK_MODE = True  → keyword-based mock, no API key needed (development).
    - MOCK_MODE = False → real Hugging Face LLM, requires HF_TOKEN in .env.

    Divya's backend import:
        from llm_service import analyze_conversation
    """
    if not conversation or not conversation.strip():
        raise ValueError("conversation must be a non-empty string.")

    if MOCK_MODE:
        return _analyze_mock(conversation)
    else:
        return _analyze_with_huggingface(conversation)


def generate_meeting_brief(memories: list) -> dict:
    """
    Generate a pre-meeting briefing from a list of customer memory strings.

    Parameters
    ----------
    memories : list of str
        Memory entries retrieved from Hindsight (Kavya's component).
        Each entry is a short string describing a past interaction fact.

    Returns
    -------
    dict with keys:
        customer_summary, key_objections, competitors, requirements,
        open_commitments, suggested_talking_points

    Example
    -------
    >>> memories = ["Rahul considers Rs. 50,000 too expensive.", "Rahul needs analytics."]
    >>> brief = generate_meeting_brief(memories)
    >>> brief["customer_summary"]
    'Rahul from ABC Technologies is an active prospect...'

    Notes
    -----
    - MOCK_MODE = True  → keyword-based mock, no API key needed (development).
    - MOCK_MODE = False → real Hugging Face LLM, requires HF_TOKEN in .env.
    - Do NOT pass Hindsight connection logic here — provide already-fetched memories.

    Divya's backend import:
        from llm_service import generate_meeting_brief
    """
    if not isinstance(memories, list) or len(memories) == 0:
        raise ValueError("memories must be a non-empty list of strings.")

    if MOCK_MODE:
        return _generate_brief_mock(memories)
    else:
        return _generate_brief_with_huggingface(memories)
