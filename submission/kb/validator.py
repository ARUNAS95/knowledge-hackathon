"""Drop any answer sentence with a number, URL or phone that is not in the evidence, the
question or the system's own text. Numbers are compared by value ("3.5%" == "3.50 %").
"""
import re

_NUM = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)
_URL = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+(?:com|org|net|gov)(?:/[^\s)\],;]*)?", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(*-])|\n+")


def numbers(text: str) -> set:
    out = set()
    for m in _NUM.finditer(text or ""):
        unit = "$" if m.group(1) else ("%" if m.group(4) else "")
        out.add((unit, round(float(m.group(2).replace(",", "") + ("." + m.group(3) if m.group(3) else "")), 4)))
    return out


def _url_key(u: str) -> str:
    u = re.sub(r"^https?://", "", u.lower()).removeprefix("www.")
    return u.rstrip("/.")


def _num_ok(n, allowed_vals: set, allowed: set, evidence_pcts: set) -> bool:
    # Percentages must come from the evidence: a rate the member typed may be wrong.
    if n[0] == "%":
        return n[1] in evidence_pcts
    return n in allowed or (n[1] in allowed_vals)


_CLAUSE = re.compile(r",\s+(?:and|but|while|whereas)\s+|,\s+(?=not\b)|;\s+|\s+but\s+")
_YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*(\d{2})\b(?!\d)")


def _expanded_years(text: str) -> set:
    """'2024-27 Model Year' -> {2027}: the writer may spell the range out as 2024-2027."""
    return {("", float(int(m.group(1)) // 100 * 100 + int(m.group(2)))) for m in _YEAR_RANGE.finditer(text)}


def validate(answer: str, evidence_texts: list[str], question: str, system_text: str = "") -> tuple[str, list[str]]:
    """Returns (cleaned answer, dropped sentences)."""
    source = "\n".join(evidence_texts) + "\n" + question + "\n" + system_text
    allowed = numbers(source) | _expanded_years(source)
    allowed_vals = {v for _, v in allowed}
    evidence_pcts = {v for u, v in numbers("\n".join(evidence_texts) + "\n" + system_text) if u == "%"}
    src_urls = {_url_key(u) for u in _URL.findall(source)}
    def ok(text: str) -> bool:
        if any(not any(_url_key(u) in s or s in _url_key(u) for s in src_urls) for u in _URL.findall(text)):
            return False
        text_wo_urls = _URL.sub("", text)            # digits inside URLs are checked as URLs
        return all(_num_ok(n, allowed_vals, allowed, evidence_pcts) for n in numbers(text_wo_urls))

    kept, dropped = [], []
    for sent in _SENT.split(answer.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if ok(sent):
            kept.append(sent)
            continue
        # Keep the parts of a failing sentence that pass ("X pays 4.10%, but there is no 9%").
        parts = _CLAUSE.split(sent.rstrip("."))
        good = [p.strip() for p in parts if p.strip() and ok(p)]
        if good and len(good) < len(parts):
            text = "; ".join(good)
            kept.append(text[0].upper() + text[1:] + ".")
            dropped.extend(p.strip() for p in parts if p.strip() and not ok(p))
        else:
            dropped.append(sent)
    return " ".join(kept), dropped
