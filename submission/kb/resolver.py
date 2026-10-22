"""Pick the evidence to answer from, among the search hits in the selected crawl.

Order of preference:
  1. tier     the row's term, balance, loan amount or model year fits the question (clear mismatches are removed)
  2. score    relevance scaled by how much this page type is trusted for the question's topic
  3. date     newer effective or post date breaks ties
"""
import re
from dataclasses import dataclass, field

from dataclasses import replace
from types import SimpleNamespace

from config import DIGEST_MIN_HITS, LINK_BOOST, RATE_TABLE_BOOST

from .metadata import extract_qualifiers
from .retrieval import tokenize

# topic -> page_type -> trust (0..1)
TRUST = {
    # /rates copies each product page's table, so the product page is the one to cite.
    "rates":     {"product": 1.0, "rates": 0.95, "home": 0.6, "document": 0.6, "contact": 0.5,
                  "about": 0.5, "page": 0.5, "calculator": 0.3, "blog": 0.3, "news": 0.25, "listing": 0.1},
    "locations": {"contact": 1.0, "home": 0.9, "about": 0.6, "product": 0.6, "page": 0.5,
                  "news": 0.4, "blog": 0.3, "document": 0.3, "rates": 0.3, "calculator": 0.1, "listing": 0.1},
    "org":       {"about": 0.9, "home": 0.85, "contact": 0.7, "news": 0.7, "product": 0.6,
                  "page": 0.6, "blog": 0.5, "document": 0.5, "rates": 0.3, "calculator": 0.1, "listing": 0.1},
    "general":   {"product": 0.9, "about": 0.85, "contact": 0.85, "home": 0.8, "rates": 0.8,
                  "page": 0.7, "blog": 0.6, "news": 0.5, "document": 0.5, "calculator": 0.4, "listing": 0.1},
}
TRUST_REASON = {
    "rates": "rate/fee question: official rate tables and product pages outrank articles",
    "locations": "hours/location question: the locations page outranks news and blogs",
    "org": "organisation fact: about/home pages and dated news footers",
    "general": "general question: product and service pages first",
}
CARRIED_FORWARD_FACTOR = 0.8
MAX_PER_PAGE = 2

_RATES = re.compile(r"\b(rates?|apy|apr|dividends?|fees?|cost|charge|interest|yield|%)", re.I)
_LOC = re.compile(r"\b(hours?|address|branch(es)?|atm|open|close[sd]?|located|location|drive[- ]?(?:up|thru|through)|lobby)\b", re.I)
_ORG = re.compile(r"\b(how many|members|employees|assets|founded|forbes|award|recogni[sz]|ranked|volunteer)\b", re.I)
# Counts stated in the "About Dupaco" footer. "How many times" asks how often an event happened,
# which the post body answers, so it is left out.
_ORG_COUNT = re.compile(r"\b(how many(?! times)|number of(?! times)|members|employees|assets|founded|branches|counties)\b", re.I)
# Who may join: only the footer names the area in text.
_MEMBERSHIP = re.compile(r"\b(eligib\w*|qualif\w*|who can (?:join|become)|charter area|membership requirements?)\b", re.I)
# Republished posts share a URL plus a trailing -2, -3.
_SERIES_SUFFIX = re.compile(r"-\d+$")
# Questions about a period, where older posts are the answer. A model year is a tier, not a period.
YEAR_QUESTION = re.compile(r"\b(?:last|this|past|previous) (?:year|month|quarter)\b|\bin (?:19|20)\d{2}\b(?![- ]model)", re.I)
# Questions about a status or a past event: an end date is the answer, not a sign of staleness.
_STATUS_Q = re.compile(r"\b(close[sd]?|closing|still open|open today|when did|ended|end(?:s|ing)?|expire[sd]?|history)\b", re.I)
EXPIRED_PENALTY = 0.3
# The member wants every option, so the writer gets more items.
LIST_QUESTION = re.compile(
    r"\b((?:what|which) (?:\w+ ){0,3}(?:do you|does dupaco) (?:offer|have)|which (?:\w+ ){0,3}(?:do you|are)|list|all (?:of )?(?:your|the)|"
    r"what are (?:the |your )?(?:\w+ ){0,3}(?:rates|options|cards|accounts|fees|hours)|rates for the)\b", re.I)
# A savings question naming no account is answered for basic Share Savings first.
_SAVINGS_Q = re.compile(r"\bsavings?\b", re.I)
_NAMED_SAVINGS = re.compile(r"share savings|money market|health|\bhsa\b|holiday|club|you name it|high[- ]yield|"
                            r"launchpad|bright|business|youth|kids?|teen|\bira\b|certificate|\bcds?\b|christmas", re.I)


def unnamed_savings(question: str) -> bool:
    return bool(_SAVINGS_Q.search(question)) and not _NAMED_SAVINGS.search(question)


_AMOUNT = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k|thousand)?", re.I)
# The member wants a link or an online route, not a description.
LINK_QUESTION = re.compile(
    r"\b(link|url|website|web ?page|online|apply|sign ?up|log ?in|where (?:can|do) i (?:see|find|go))\b", re.I)
# Words every link question shares; they say nothing about which link is wanted.
_LINK_GENERIC = set(tokenize("link url website web page online http https www com dupaco see find go use "
                              "apply sign up open start become register enroll"))


def topic_of(question: str) -> str:
    if _ORG.search(question):
        return "org"
    if _LOC.search(question) and not _RATES.search(question):
        return "locations"
    if _RATES.search(question):
        return "rates"
    return "general"


@dataclass
class QuestionTiers:
    qual: dict
    amounts: list = field(default_factory=list)

    @classmethod
    def parse(cls, question: str, intent: dict = None) -> "QuestionTiers":
        amounts = []
        for m in _AMOUNT.finditer(question):
            v = float(m.group(1).replace(",", ""))
            amounts.append(v * 1000 if m.group(2) else v)
        qual = extract_qualifiers(question)
        # The LLM intent fills gaps the regexes miss; values found in the question text win.
        intent = intent or {}
        if intent.get("term_months") and not qual.get("term_months"):
            qual["term_months"] = [int(intent["term_months"])]
        if intent.get("amount") and not amounts:
            amounts = [float(intent["amount"])]
        if intent.get("model_year") and not qual.get("model_year"):
            y = int(intent["model_year"])
            qual["model_year"] = (y, y)
        if intent.get("jumbo") is True:
            qual["jumbo"] = True
        if intent.get("purchase_or_refinance"):
            qual[intent["purchase_or_refinance"]] = True
        return cls(qual, amounts)

    @property
    def names_tier(self) -> bool:
        q = self.qual
        return bool(q.get("term_months") or self.amounts or q.get("model_year") or q.get("jumbo")
                    or q.get("purchase") or q.get("refinance"))


def _in(lo, hi, v) -> bool:
    return (lo is None or v >= lo - 0.01) and (hi is None or v <= hi + 0.01)


def tier_score(q: QuestionTiers, occ) -> float:
    """>0 fits, 0 neutral, <=-2 clear mismatch. Only rows carry tier values."""
    if occ.chunk_type != "row":
        return 0.0
    oq = occ.meta["qualifiers"]
    s = 0.0
    if q.qual.get("term_months") and oq.get("term_months"):
        s += 1 if set(q.qual["term_months"]) & set(oq["term_months"]) else -2
    if q.amounts and oq.get("balance"):
        s += 1 if any(_in(*oq["balance"], a) for a in q.amounts) else -2
    if q.amounts and oq.get("loan_amount"):
        s += 0.5 if any(_in(*oq["loan_amount"], a) for a in q.amounts) else -2
    if q.qual.get("model_year") and oq.get("model_year"):
        y = q.qual["model_year"][1]
        s += 1 if _in(*oq["model_year"], y) else -2
    if q.qual.get("jumbo") != oq.get("jumbo", False):
        s -= 2 if q.qual.get("jumbo") else 0.5      # asked jumbo -> must be jumbo; else prefer regular
    if q.qual.get("special") and not oq.get("special"):
        s -= 1
    for k in ("purchase", "refinance"):
        if q.qual.get(k) and not oq.get(k) and (oq.get("purchase") or oq.get("refinance")):
            s -= 2
    return s


def _date(occ) -> str:
    d = occ.meta["dates"]
    for k in ("effective", "post"):
        if d.get(k):
            return d[k]["date"]
    return ""


@dataclass
class Evidence:
    rec: object
    occ: object                   # representative occurrence (most trusted mirror)
    urls: list
    relevance: float
    tier: float
    trust: float
    date: str
    carried: bool
    reasons: list

    @property
    def text(self) -> str:
        if self.occ.chunk_type == "row" and self.occ.meta.get("notes"):
            return self.rec.text + "\n\n" + self.occ.meta["notes"]
        return self.rec.text


def resolve(question: str, hits, store, snapshot_id: str, max_items: int = 5, intent: dict = None,
            trace: dict = None) -> list[Evidence]:
    """hits: [(record, relevance)] best first. trace, if given, records every decision."""
    if not hits:
        return []
    steps = trace.setdefault("steps", []) if trace is not None else None

    def log(rec, stage, **info):
        if steps is not None:
            o = (store.occurrences_in(rec, snapshot_id) or rec.occurrences)[0] if getattr(rec, "occurrences", None) else None
            steps.append({"stage": stage, "content_id": rec.content_id, "type": rec.chunk_type,
                          "url": o.url_norm if o else "", **info})
    intent = intent or {}
    topic = intent.get("topic") or topic_of(question)
    trust_table = TRUST[topic]
    q = QuestionTiers.parse(question, intent)
    # A matching row is replaced by its whole table so the writer sees headers, dates and footnotes.
    widen = True
    if intent.get("wants_all") or (not intent and LIST_QUESTION.search(question)):
        max_items = max(max_items, 8)
    def boosted(rec, rel):
        if topic == "rates" and rec.chunk_type in ("table", "row") and re.search(r"\d%|\$\d", rec.text):
            return rel + RATE_TABLE_BOOST
        # Organisation counts live in the "About Dupaco" footer under each news post.
        if topic == "org" and rec.chunk_type == "footer" and _ORG_COUNT.search(question):
            return rel + RATE_TABLE_BOOST
        if (link_words and any(o.url_norm in link_pages for o in store.occurrences_in(rec, snapshot_id))
                and _link_match(rec, link_words)):
            return rel + LINK_BOOST
        return rel
    # The link boost only applies to pages search already ranked high, so it picks a section, not a page.
    link_words = set(tokenize(question)) - _LINK_GENERIC if LINK_QUESTION.search(question) else set()
    link_pages = set(list(dict.fromkeys(o.url_norm for rec, _ in hits for o in store.occurrences_in(rec, snapshot_id)[:1]))[:5])
    digest_url = _dominant_page(hits, store, snapshot_id) if topic in ("general", "org") else None
    if trace is not None:
        trace.update(topic=topic, trust_table=trust_table, tiers=q.qual, amounts=q.amounts, max_items=max_items,
                     link_words=sorted(link_words), link_pages=sorted(link_pages), digest_page=digest_url,
                     membership_question=bool(_MEMBERSHIP.search(question)))
    if topic == "rates":
        n0 = len(hits)
        hits = _expand_rate_pages(hits, store, snapshot_id)
        for rec, rel in hits[n0:]:
            log(rec, "added", why="same-page rate statement (rate question)", relevance=round(rel, 3))
    boosted_hits = []
    for rec, rel in hits:
        b = boosted(rec, rel)
        if b != rel:
            log(rec, "boost", why=f"+{b - rel:.2f}", relevance=round(rel, 3), boosted=round(b, 3))
        boosted_hits.append((rec, b))
    hits = boosted_hits
    best_rel = max(r for _, r in hits)
    out = []
    seen = set()
    for rec, rel in hits:
        occs = store.occurrences_in(rec, snapshot_id)
        if not occs:
            continue
        if widen and rec.chunk_type == "row":
            o = occs[0]
            if tier_score(q, o) <= -2:
                log(rec, "removed", why=f"row tier does not match the question ({o.meta['qualifiers']})")
                continue
            table = store.table_index.get((o.snapshot_id, o.table_id))
            if table is not None:
                log(rec, "widened", why=f"row -> its whole table #{table.content_id}")
                rec = table
                occs = store.occurrences_in(rec, snapshot_id) or occs
        if rec.content_id in seen:
            log(rec, "removed", why="duplicate (already a candidate)")
            continue
        seen.add(rec.content_id)
        # The same chunk on several URLs becomes one item citing the most trusted, then newest, URL.
        occ = max(occs, key=lambda o: (trust_table.get(o.page_type, 0.5), _date(o)))
        carried = occ.snapshot_id != snapshot_id
        trust = trust_table.get(occ.page_type, 0.5) * (CARRIED_FORWARD_FACTOR if carried else 1.0)
        # Old posts lose trust.
        trust -= 0.15 * sum(1 for w in occ.meta["warnings"] if w.startswith("post is"))
        # A past end date likely means an expired rate, unless the question is about that end date.
        if topic == "rates" and not _STATUS_Q.search(question) and any(
                w.startswith("mentions an end date") for w in occ.meta["warnings"]):
            trust -= EXPIRED_PENALTY
            log(rec, "penalised", why=f"end date before this crawl (rate question): trust -{EXPIRED_PENALTY}")
        t = tier_score(q, occ)
        if t <= -2:
            log(rec, "removed", why=f"tier mismatch {occ.meta['qualifiers']}")
            continue
        reasons = [f"{occ.page_type} page ({TRUST_REASON[topic]})"]
        if carried:
            reasons.append(f"page missing from this crawl; using its {occ.snapshot_id} copy")
        reasons += occ.meta["warnings"]
        urls = [o.url for o in sorted(occs, key=lambda o: (-trust_table.get(o.page_type, 0.5), o.url))]
        out.append(Evidence(rec, occ, list(dict.fromkeys(urls)), rel, t, trust, _date(occ), carried, reasons))
    if not out:
        return []
    def dropped(before, after, why):
        keep = {id(e) for e in after}
        for e in before:
            if id(e) not in keep:
                log(e.rec, "removed", why=why)

    step = _drop_superseded_tables(out)
    dropped(out, step, "older table with the same products (superseded by a newer effective date)")
    out, step = step, _merge_same_text(step)
    dropped(out, step, "same text as a better-ranked item (its URLs were merged into that item)")
    out = step
    out.sort(key=lambda e: (-min(e.tier, 1), -round(e.relevance * (0.7 + 0.3 * e.trust), 2), _neg(e.date)))
    # Skipped for hours (each post is a different branch) and period questions (older posts are the answer).
    if topic != "locations" and not YEAR_QUESTION.search(question):
        step = _newest_in_series(out)
        dropped(out, step, "older post of a republished series (newer -N version kept)")
        out = step
    if digest_url:
        step = _with_digest(out, digest_url, store, snapshot_id)
        dropped(out, step, "prose section merged into the page digest")
        out = step
    step = _merge_same_page(out)
    dropped(out, step, "merged into one item with the same page's other section, in page order")
    out = step
    if trace is not None:
        trace["ranked"] = [{"content_id": e.rec.content_id, "type": e.occ.chunk_type, "url": e.occ.url_norm,
                            "page_type": e.occ.page_type, "relevance": round(e.relevance, 3), "tier": e.tier,
                            "trust": round(e.trust, 3), "score": round(e.relevance * (0.7 + 0.3 * e.trust), 3),
                            "date": e.date, "reasons": e.reasons, "text": e.text} for e in out]
    picked, per_page = [], {}
    for e in out:
        page = e.occ.url_norm
        if per_page.get(page, 0) >= MAX_PER_PAGE:
            log(e.rec, "skipped", why=f"page already has {MAX_PER_PAGE} items")
            continue
        per_page[page] = per_page.get(page, 0) + 1
        picked.append(e)
        if len(picked) == max_items:
            break
    # The footer states the service area and counts, but other posts can outrank it.
    if _MEMBERSHIP.search(question) or (topic == "org" and _ORG_COUNT.search(question)):
        footer = _newest_footer(store, snapshot_id, trust_table)
        if footer is not None and all(e.rec.content_id != footer.rec.content_id for e in picked):
            picked.append(footer)
            log(footer.rec, "added", why="membership/count question: newest About Dupaco footer")
    if topic == "rates" and unnamed_savings(question):
        share = _share_savings_table(store, snapshot_id, trust_table)
        if share is not None and all(e.rec.content_id != share.rec.content_id for e in picked):
            picked = [share] + picked[: max_items - 1]
            log(share.rec, "added", why="savings question naming no account: basic Share Savings table first")
    if topic == "locations" and _HOURS_Q.search(question):
        hours = _branch_schedules(store, snapshot_id, trust_table)
        if hours is not None:
            picked.append(hours)
            log(hours.rec, "added", why=f"hours question: every branch schedule on the site ({hours.reasons[0]})")
    return picked


def _newest_in_series(items: list) -> list:
    """Of a republished series of posts, keep only the newest."""
    newest: dict[str, str] = {}
    for e in items:
        if e.occ.page_type in ("news", "blog"):
            key = _SERIES_SUFFIX.sub("", e.occ.url_norm)
            newest[key] = max(newest.get(key, ""), e.date)
    return [e for e in items if e.occ.page_type not in ("news", "blog")
            or e.date == newest[_SERIES_SUFFIX.sub("", e.occ.url_norm)]]


def _merge_same_page(items: list) -> list:
    """Join a page's top prose items into one item, in page order, so the writer reads them in context."""
    groups: dict[str, list] = {}
    for e in items:
        if e.occ.chunk_type not in ("table", "row", "digest"):
            groups.setdefault(e.occ.url_norm, []).append(e)
    merged = {}
    for url, es in groups.items():
        if len(es) < 2:
            continue
        top = es[:MAX_PER_PAGE]
        in_page = sorted(top, key=lambda e: e.occ.chunk_index)
        text = "\n\n".join(e.text for e in in_page)
        rec = SimpleNamespace(text=text, content_id=top[0].rec.content_id, chunk_type="merged")
        occ = replace(in_page[0].occ, chunk_type="merged", text=text)
        urls = list(dict.fromkeys(u for e in top for u in e.urls))
        merged[url] = (id(top[0]), replace(top[0], rec=rec, occ=occ, urls=urls),
                       {id(e) for e in es})
    out = []
    for e in items:
        m = merged.get(e.occ.url_norm)
        if m is None or id(e) not in m[2]:
            out.append(e)
        elif id(e) == m[0]:
            out.append(m[1])
    return out


_HOURS_Q = re.compile(r"\b(hours?|lobby|drive[- ]?(?:up|thru|through)|open on|closed on|when is .{0,40}open)\b", re.I)
_HOURS_BLOCK = re.compile(r"(?:Branch hours are:\s*)?\*\*Lobby\*\*.*?(?:\*\*Drive[- ]?up\*\*.*?Sunday[^\n]*|$)", re.S | re.I)


def _branch_schedules(store, snapshot_id: str, trust_table: dict):
    """One item listing every distinct branch schedule, since hours only appear in branch-opening posts."""
    cache = store.__dict__.setdefault("_hours_cache", {})
    if snapshot_id not in cache:
        blocks: dict[str, list] = {}              # normalised schedule -> [(occ, intro, text)]
        for rec in store.records:
            if snapshot_id not in rec.visible_in or "**Lobby**" not in rec.text:
                continue
            m = _HOURS_BLOCK.search(rec.text)
            if not m:
                continue
            sched = re.sub(r"\n\s*\n", "\n", m.group(0).replace("Branch hours are:", "")).strip()
            before = rec.text[: m.start()].strip().split("\n")[-1]
            intro = before if re.search(r"branch|learning lab|phone number", before, re.I) else ""
            occ = store.occurrences_in(rec, snapshot_id)[0]
            key = re.sub(r"[\s*–—-]+", " ", sched.lower())
            blocks.setdefault(key, []).append((occ, intro, sched))
        cache[snapshot_id] = blocks
    blocks = cache[snapshot_id]
    if not blocks:
        return None
    # Lobby and drive-up schedules are listed separately so the writer doesn't mix them up.
    by_kind: dict[str, dict] = {"Lobby": {}, "Drive-up": {}}
    occs = []
    for entries in blocks.values():
        lobby, _, drive = (re.split(r"(\*\*Drive[- ]?up\*\*)", entries[0][2], maxsplit=1, flags=re.I) + ["", ""])[:3]
        for kind, part in (("Lobby", lobby), ("Drive-up", drive)):
            lines = [l.strip() for l in part.replace("**Lobby**", "").split("\n") if l.strip()]
            if not lines:
                continue
            key = re.sub(r"[\s*–—-]+", " ", " ".join(lines).lower())
            slot = by_kind[kind].setdefault(key, (lines, []))
            slot[1].extend(intro or o.title for o, intro, _ in entries)
        occs += [o for o, _, _ in entries]
    parts = []
    for kind, scheds in by_kind.items():
        if not scheds:
            continue
        parts.append(f"{kind} hours: the site lists {len(scheds)} different {kind.lower()} schedule"
                     f"{'s' if len(scheds) > 1 else ''}.")
        for n, (lines, names) in enumerate(scheds.values(), 1):
            parts.append(f"{kind} schedule {n} ({'; '.join(dict.fromkeys(names))}):\n" + "\n".join(lines))
    text = "\n\n".join(parts)
    base = occs[0]
    meta = dict(base.meta, dates={"post": None, "effective": None, "footnote": {}, "expiration": None, "event": []},
                warnings=[])
    occ = replace(base, chunk_type="hours", title="Branch hours listed on the site", heading_path=[], label="",
                  text=text, meta=meta)
    rec = SimpleNamespace(text=text, content_id=-2, chunk_type="hours")
    return Evidence(rec, occ, list(dict.fromkeys(o.url for o in occs)), 0.0, 0.0, trust_table.get("news", 0.5), "",
                    False, [f"{len(blocks)} distinct branch schedules"])


def _newest_footer(store, snapshot_id: str, trust_table: dict):
    """The newest post's "About Dupaco" footer in this crawl, or None."""
    cache = store.__dict__.setdefault("_footer_cache", {})
    if snapshot_id not in cache:
        best = None
        for rec in store.records:
            if rec.chunk_type != "footer" or "serves residents" not in rec.text:
                continue
            for o in store.occurrences_in(rec, snapshot_id):
                if best is None or (_date(o), o.url) > (_date(best[1]), best[1].url):
                    best = (rec, o)
        cache[snapshot_id] = best
    if cache[snapshot_id] is None:
        return None
    rec, occ = cache[snapshot_id]
    return Evidence(rec, occ, [occ.url], 0.0, 0.0, trust_table.get(occ.page_type, 0.5), _date(occ),
                    False, ["organisation footer: names the charter area"])


def _share_savings_table(store, snapshot_id: str, trust_table: dict):
    """The Share Savings rate table in this crawl (most trusted page, newest date), or None."""
    cache = store.__dict__.setdefault("_share_cache", {})
    if snapshot_id not in cache:
        best = None
        for rec in store.records:
            if rec.chunk_type != "table" or "| Share Savings |" not in rec.text:
                continue
            for o in store.occurrences_in(rec, snapshot_id):
                # On equal trust, prefer the shortest URL (the account's own page).
                k = (trust_table.get(o.page_type, 0.5), _date(o), -len(o.url))
                if best is None or k > best[0]:
                    best = (k, rec, o)
        cache[snapshot_id] = best
    if cache[snapshot_id] is None:
        return None
    _, rec, occ = cache[snapshot_id]
    urls = [o.url for o in sorted(store.occurrences_in(rec, snapshot_id),
                                  key=lambda o: (-trust_table.get(o.page_type, 0.5), o.url))]
    return Evidence(rec, occ, list(dict.fromkeys(urls)), 0.0, 0.0, trust_table.get(occ.page_type, 0.5), _date(occ),
                    False, ["basic savings account: answered first when the question names no account"])


def _link_match(rec, words: set) -> bool:
    """True if a link in this chunk shares a word with the question."""
    for link in rec.occurrences[0].meta.get("links", []):
        if words & set(tokenize(f"{link['text']} {re.sub(r'[^a-z0-9]+', ' ', link['url'].lower())}")):
            return True
    return False


def _dominant_page(hits, store, snapshot_id):
    """The page holding at least DIGEST_MIN_HITS of the top 10 hits, else None."""
    counts: dict[str, int] = {}
    for rec, _ in hits[:10]:
        occs = store.occurrences_in(rec, snapshot_id)
        if occs:
            counts[occs[0].url_norm] = counts.get(occs[0].url_norm, 0) + 1
    url, n = max(counts.items(), key=lambda x: x[1], default=(None, 0))
    return url if n >= DIGEST_MIN_HITS else None


def _with_digest(items: list, url: str, store, snapshot_id) -> list:
    """Replace the page's prose items with one item holding all its prose, in page order."""
    first = next((i for i, e in enumerate(items) if e.occ.url_norm == url), None)
    if first is None:
        return items
    base = items[first]
    prose = [r for r in store.page_index.get((base.occ.snapshot_id, url), []) if r.chunk_type not in ("table", "row")]
    if len(prose) <= MAX_PER_PAGE:
        return items
    parts, last = [], None
    for r in prose:
        o = next((o for o in r.occurrences if o.url_norm == url and o.snapshot_id == base.occ.snapshot_id), r.occurrences[0])
        head = o.heading_path[-1] if o.heading_path else ""
        parts.append(f"## {head}\n{r.text}" if head and head != last else r.text)
        last = head
    text = "\n\n".join(parts)
    rec = SimpleNamespace(text=text, content_id=-1, chunk_type="digest")
    occ = replace(base.occ, chunk_type="digest", heading_path=[], label="", text=text)
    page_items = [e for e in items if e.occ.url_norm == url]
    digest = replace(base, rec=rec, occ=occ, relevance=max(e.relevance for e in page_items), tier=0)
    rest = [e for e in items if e.occ.url_norm != url or e.occ.chunk_type in ("table", "row")]
    return rest[:first] + [digest] + rest[first:]


_RATE_TEXT = re.compile(r"\d(?:\.\d+)?%\s*(?:[-–]\s*\d+(?:\.\d+)?%\s*)?(?:APR|APY|\b)", re.I)


def _expand_rate_pages(hits, store, snapshot_id, top_pages: int = 4):
    """Add rate statements from the top product pages, which can rank low on their own."""
    have = {rec.content_id for rec, _ in hits}
    best: dict[str, float] = {}
    for rec, rel in hits:
        for o in store.occurrences_in(rec, snapshot_id)[:1]:
            if o.page_type in ("product", "rates") and len(best) < top_pages and o.url_norm not in best:
                best[o.url_norm] = rel
    extra = []
    for url, rel in best.items():
        for rec in store.page_index.get((snapshot_id, url), []):
            if rec.content_id not in have and _RATE_TEXT.search(rec.text):
                have.add(rec.content_id)
                extra.append((rec, rel * 0.9))
    return hits + extra


def _merge_same_text(items: list) -> list:
    """Keep one copy of identical text and give it every page's URL."""
    by_text: dict[str, Evidence] = {}
    out = []
    for e in sorted(items, key=lambda e: -e.relevance * (0.7 + 0.3 * e.trust)):
        key = re.sub(r"\s+", " ", e.rec.text).strip()
        if key in by_text:
            keep = by_text[key]
            keep.urls += [u for u in e.urls if u not in keep.urls]
            continue
        by_text[key] = e
        out.append(e)
    return out


def _row_names(e) -> set:
    """First-column cells of a table, normalised."""
    names = set()
    for line in e.rec.text.split("\n")[2:]:
        if line.strip().startswith("|"):
            cell = line.strip().strip("|").split("|")[0]
            cell = re.sub(r"\$[\d,]+(?:\.\d+)?\s*or less|[^a-z0-9 -]", "", cell.lower()).strip()
            if cell:
                names.add(cell)
    return names


def _drop_superseded_tables(items: list) -> list:
    """Drop a rate table when a newer table lists the same products."""
    tables = [e for e in items if e.occ.chunk_type in ("table", "row") and e.date]
    drop = set()
    for a in tables:
        na = _row_names(a)
        for b in tables:
            if a is b or b.date <= a.date or not na:
                continue
            nb = _row_names(b)
            if nb and len(na & nb) / min(len(na), len(nb)) >= 0.5:
                drop.add(id(a))
                break
    return [e for e in items if id(e) not in drop]


def _neg(d: str) -> str:
    """Sort newer ISO dates first inside an ascending sort key."""
    return "".join(chr(0x7F - ord(c)) for c in d) if d else "\x7f"
