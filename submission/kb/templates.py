"""Fixed answer texts and templates used when the LLM is unavailable.
Refusals contain no digits, since a refusal must not state a number.
"""
import re

REFUSAL = "I don't know. The Dupaco website doesn't list that, so please contact Dupaco directly to confirm."
TECHNICAL = "I can't confirm that right now because of a technical issue. Please contact Dupaco directly."

_FOOTNOTE = re.compile(r"(\d%)\d\b")       # "4.10%2" -> "4.10%"


def as_of_prefix(taken_on: str) -> str:
    return f"According to the latest available snapshot ({taken_on}), "


def from_evidence(e) -> str:
    """State the top evidence plainly; a table row becomes 'Row: Column value, ...'."""
    o = e.occ
    eff = o.meta["dates"].get("effective")
    eff_txt = f" (rates effective {eff['evidence'].split('effective as of ')[-1].split()[0]})" if eff and "effective as of" in eff["evidence"] else ""
    if o.chunk_type == "row" and o.meta.get("qualifiers") is not None:
        lines = [l for l in e.rec.text.split("\n") if l.strip().startswith("|")]
        if len(lines) >= 2:
            header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
            row = [c.strip() for c in lines[-1].strip().strip("|").split("|")]
            parts = [f"{h} {v}" for h, v in zip(header[1:], row[1:]) if v and not h.startswith("*")]
            return _FOOTNOTE.sub(r"\1", f"{row[0]}: {', '.join(parts)}{eff_txt}.")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", e.text)          # markdown links -> text
    text = re.sub(r"\[Image: [^\]]*\]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _FOOTNOTE.sub(r"\1", text[:500])
