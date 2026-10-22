"""Extract per-chunk metadata: dates with their source text, tier qualifiers as numeric
ranges, values without footnote markers, links and entities. The resolver uses these.
"""
import re
from datetime import date, datetime
from typing import Optional

from .urls import clean_url

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MDY_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MDY_LONG = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
_EFFECTIVE = re.compile(rf"effective(?: as of| on| beginning)?\s+((?:\d{{1,2}}/\d{{1,2}}/\d{{4}})|(?:(?:{_MONTHS})\.? \d{{1,2}},? \d{{4}}))", re.I)
_FOOTNOTE_EFFECTIVE = re.compile(r"(?:^|\n)(\d)(?=[A-Z])[^\n]*?effective as of (\d{1,2}/\d{1,2}/\d{4})")
_EXPIRES = re.compile(rf"(?:expires?|ends?|through|until|valid through|offer ends|close[sd]?|closing)\s+(?:on\s+)?((?:\d{{1,2}}/\d{{1,2}}/\d{{4}})|(?:(?:{_MONTHS})\.? \d{{1,2}},? \d{{4}}))", re.I)

# "4.10%2" is 4.10% with footnote 2.
_PCT = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,3})?)\s?%(\d)?(?!\d)")
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?(\+)?")
_PHONE = re.compile(r"\b(?:1[.-])?\(?(\d{3})\)?[.\- ](\d{3})[.\-](\d{4})\b")
_ROUTING = re.compile(r"\b(\d{9})\b")
_URL = re.compile(r"https?://[^\s)\]]+")
# Markdown links with visible text; images and in-page anchors are skipped.
_MD_LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^)\s]+)\)")

_TERM = re.compile(r"\b(\d{1,3})[- ]?(month|mo\.?|year|yr)s?\b", re.I)
_RANGE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:–|-|to)\s*\$\s?([\d,]+(?:\.\d+)?)")
_PLUS = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*\+")
_OR_LESS = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*or less", re.I)
# "2024-27 Model Year", "2025 model-year vehicle", "2016 Model Year (and older)"
_MODEL_YEAR = re.compile(r"\b((?:19|20)\d{2})(?:\s*(?:–|-|to)\s*(\d{2,4}))?\s*(?:model|vehicle)[- ]?(?:year)?(\s*\(and older\))?", re.I)


def _to_iso(s: str) -> Optional[str]:
    s = s.strip().replace(".", "")
    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _snippet(text: str, start: int, end: int, pad: int = 60) -> str:
    return text[max(0, start - pad): end + pad].replace("\n", " ").strip()


def extract_dates(text: str, post_date: Optional[str]) -> dict:
    out: dict = {"post": None, "effective": None, "footnote": {}, "expiration": None, "event": []}
    if post_date:
        out["post"] = {"date": post_date, "evidence": "post date in page header"}
    effs = []
    for m in _EFFECTIVE.finditer(text):
        iso = _to_iso(m.group(1))
        if iso:
            effs.append({"date": iso, "evidence": _snippet(text, m.start(), m.end())})
    for m in _FOOTNOTE_EFFECTIVE.finditer(text):
        iso = _to_iso(m.group(2))
        if iso:
            out["footnote"][m.group(1)] = {"date": iso, "evidence": _snippet(text, m.start(), m.end())}
    # Prefer the table's own "effective" line over dates in footnotes.
    plain = [e for e in effs if e["date"] not in {f["date"] for f in out["footnote"].values()}] or effs
    if plain:
        out["effective"] = max(plain, key=lambda e: e["date"])
    m = _EXPIRES.search(text)
    if m and _to_iso(m.group(1)):
        out["expiration"] = {"date": _to_iso(m.group(1)), "evidence": _snippet(text, m.start(), m.end())}
    for rx in (_MDY_LONG, _MDY_SLASH):
        for m in rx.finditer(text):
            iso = _to_iso(m.group(0))
            if iso and iso not in {d["date"] for d in out["event"]}:
                out["event"].append({"date": iso, "evidence": _snippet(text, m.start(), m.end(), 40)})
    return out


def extract_qualifiers(text: str) -> dict:
    """Tier qualifiers from a row/label, as numeric ranges (None = open-ended)."""
    q: dict = {}
    terms = []
    for m in _TERM.finditer(text):
        n = int(m.group(1))
        terms.append(n * 12 if m.group(2).lower().startswith("y") else n)
    if terms:
        q["term_months"] = sorted(set(terms))
    r = _RANGE.search(text)
    if r:
        q["balance"] = (_num(r.group(1)), _num(r.group(2)))
    elif _PLUS.search(text):
        q["balance"] = (_num(_PLUS.search(text).group(1)), None)
    ol = _OR_LESS.search(text)
    if ol:
        q["loan_amount"] = (None, _num(ol.group(1)))
    my = _MODEL_YEAR.search(text)
    if my:
        lo = int(my.group(1))
        hi = my.group(2)
        hi = int(hi) if hi and len(hi) == 4 else (lo // 100 * 100 + int(hi) if hi else lo)
        q["model_year"] = (None if my.group(3) else lo, hi)
    low = text.lower()
    q["jumbo"] = "jumbo" in low
    if "purchase" in low:
        q["purchase"] = True
    if "refinanc" in low:
        q["refinance"] = True
    if re.search(r"\bintro(ductory)?\b", low):
        q["intro"] = True
    if "special" in low:
        q["special"] = True
    return q


def extract_values(text: str) -> list[dict]:
    vals = []
    for m in _PCT.finditer(text):
        vals.append({"raw": m.group(0), "value": float(m.group(1)), "unit": "%", "footnote": m.group(2)})
    for m in _MONEY.finditer(text):
        v = _num(m.group(1)) + (int(m.group(2)) / 100 if m.group(2) else 0)
        vals.append({"raw": m.group(0), "value": v, "unit": "$", "footnote": None})
    return vals


def extract_entities(text: str) -> dict:
    ents = {}
    phones = sorted({"-".join(m.groups()) for m in _PHONE.finditer(text)})
    if phones:
        ents["phones"] = phones
    routing = sorted({m.group(1) for m in _ROUTING.finditer(text) if "routing" in text.lower()})
    if routing:
        ents["routing_numbers"] = routing
    urls = sorted(set(_URL.findall(text)))
    if urls:
        ents["urls"] = urls
    return ents


def extract_links(text: str) -> list[dict]:
    """Markdown links as [{'text', 'url'}] in page order, tracking parameters removed."""
    out, seen = [], set()
    for m in _MD_LINK.finditer(text):
        url = clean_url(m.group(2))
        if url not in seen:
            seen.add(url)
            out.append({"text": m.group(1).strip(), "url": url})
    return out


def occurrence_metadata(chunk, page) -> dict:
    """Everything the resolver needs about one chunk as it appeared on one page in one crawl."""
    full = chunk.text + ("\n\n" + chunk.notes if chunk.chunk_type == "row" and chunk.notes else "")
    # Use the row and label, not the header, so a header word doesn't tag every row.
    if chunk.chunk_type == "row":
        qual_src = chunk.label + " " + chunk.text.split("\n")[-1]
    else:
        qual_src = chunk.label + " " + " ".join(chunk.heading_path[-1:])
    dates = extract_dates(full, page.post_date)
    warnings = []
    exp = dates["expiration"]
    if exp and exp["date"] < page.taken_on:
        warnings.append(f"mentions an end date {exp['date']} before this crawl ({page.taken_on})")
    if page.page_type in ("blog", "news") and page.post_date:
        age = (date.fromisoformat(page.taken_on) - date.fromisoformat(page.post_date)).days
        if age > 365:
            warnings.append(f"post is {age // 365} year(s) old")
    return {
        "dates": dates,
        "qualifiers": extract_qualifiers(qual_src),
        "product": chunk.label or (chunk.heading_path[-1] if chunk.heading_path else page.title),
        "values": extract_values(chunk.text),
        "entities": extract_entities(chunk.text),
        "links": extract_links(chunk.text),
        "warnings": warnings,
        "notes": chunk.notes if chunk.chunk_type == "row" else "",
    }
