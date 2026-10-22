"""Turn the member's question into a structured intent with an LLM.
Catches wording the regex rules miss; the regex rules remain the fallback.
"""
import re

from .llm import LLMError

SYSTEM = """You turn a credit-union member's question into JSON for a search system.
Return only JSON with these keys:
{"topic": "rates" | "locations" | "org" | "general",
 "product": short product or subject name using credit-union wording (e.g. "term share certificate", "share savings", "30-year fixed mortgage", "auto loan", "Platinum Visa", "branch lobby hours"),
 "metric": what value is asked for (e.g. "APY", "dividend rate", "APR", "rate", "fee", "hours", "address", "phone", "url", "count", "date", "eligibility", "description"),
 "term_months": number or null,
 "amount": number or null (balance or loan amount in dollars),
 "model_year": number or null,
 "jumbo": true | false | null,
 "purchase_or_refinance": "purchase" | "refinance" | null,
 "wants_all": true if the member asks for every tier/option/product (e.g. "what are your mortgage rates", "what credit cards do you offer"), else false,
 "search_query": a short keyword query (synonyms welcome, e.g. "CD" -> "certificate")}
topic: rates = rates, APY, APR, fees, dividends; locations = branches, hours, addresses, ATMs; org = facts about the credit union itself (counts, awards, history, employees); general = everything else.
Use null when the question does not say. Do not answer the question."""


def parse_intent(llm, question: str) -> dict:
    """Returns {} when the LLM is unavailable or fails."""
    if not llm or not llm.available:
        return {}
    try:
        intent = llm.chat_json(SYSTEM, f"Question: {question}", max_tokens=200)
    except LLMError:
        return {}
    return _clean(intent)


def _clean(i: dict) -> dict:
    out = {}
    if i.get("topic") in ("rates", "locations", "org", "general"):
        out["topic"] = i["topic"]
    for k in ("product", "metric", "search_query"):
        if isinstance(i.get(k), str) and i[k].strip():
            out[k] = i[k].strip()[:120]
    for k in ("term_months", "amount", "model_year"):
        v = i.get(k)
        if isinstance(v, str):
            v = re.sub(r"[^\d.]", "", v)
            v = float(v) if v else None
        if isinstance(v, (int, float)) and v > 0:
            out[k] = v
    if isinstance(i.get("jumbo"), bool):
        out["jumbo"] = i["jumbo"]
    if i.get("purchase_or_refinance") in ("purchase", "refinance"):
        out["purchase_or_refinance"] = i["purchase_or_refinance"]
    out["wants_all"] = bool(i.get("wants_all"))
    return out
