"""Find rate-table rows (product + tier + column) that show different values on different pages
of the same crawl. One conflict per stale page and product, not per row.
Winner: newest effective date, then page trust, then number of pages.
"""
import re
from collections import defaultdict

from contract import Conflict, Source

from .resolver import TRUST, _date

_TRUST = TRUST["rates"]
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_SKIP_COL = re.compile(r"min|deposit|open|payment|fee", re.I)
_UNIT = re.compile(r"%\s*(APR|APY)\d*\b", re.I)
_MODEL_YEAR = re.compile(r"\b((?:19|20)\d{2})(?:\s*-\s*\d{2,4})?\s+model year\b(?!\s*\(and older\))")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"(\d),(\d)", r"\1\2", re.sub(r"\.00\b", "", s))
    return re.sub(r"\s+", " ", s).strip()


def _clean_value(cell: str) -> str:
    """'4.10%2' -> '4.10%', '5.39% APR1' -> '5.39% APR' (footnote markers), whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"(\d%(?:\s*AP[RY])?)\d\b", r"\1", cell)).strip()


def _path(url_norm: str) -> str:
    return url_norm.split("dupaco.com", 1)[-1] or "/"


def _audience(url_norm: str) -> str:
    """Personal and business products are different facts, never a conflict."""
    return "business" if _path(url_norm).startswith("/business") else "personal"


def _year_ranks(names) -> dict:
    """Model-year buckets roll forward each year ('2023-26' becomes '2024-27'), so match them by
    position: the newest bucket in each table is 'model year #0'."""
    starts = sorted({int(m.group(1)) for n in names for m in [_MODEL_YEAR.search(n)] if m}, reverse=True)
    return {y: i for i, y in enumerate(starts)}


def _relabel(cell: str, ranks: dict) -> str:
    m = _MODEL_YEAR.search(cell)
    return _MODEL_YEAR.sub(f"model year #{ranks[int(m.group(1))]}", cell) if m and int(m.group(1)) in ranks else cell


def _metric(col: str, value: str) -> str:
    """The unit in the cell ('5.34% APR') beats the column header, which some pages get wrong."""
    m = _UNIT.search(value)
    return m.group(1).lower() if m else _norm(re.sub(r"\d+$", "", col))


_BARE_NUM = re.compile(r"^\d+(?:\.\d+)?$")
SAME_TABLE = 0.5   # share of row names two tables must have in common to count as one table


def _col_kind(col: str, value: str) -> str:
    """'%' cells are values; bare numbers ('0' points) are values written without '%'."""
    if "%" in value:
        return _metric(col, value)
    return _norm(col) if _BARE_NUM.match(value.replace(",", "")) else "tier"


def _row_names(store, o) -> set:
    table = store.table_index.get((o.snapshot_id, o.table_id))
    if table is None:
        return set()
    return {_norm(_cells(l)[0]) for l in table.text.split("\n")[2:] if l.strip().startswith("|")}


_LABEL_STOP = {"rates", "rate", "loans", "loan", "and", "the", "for", "your", "our", "dupaco"}


def _label_words(label: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]+", label.lower()) if w not in _LABEL_STOP)


def _jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _same_tables(tables: dict) -> dict:
    """Group tables whose row names mostly overlap: a page that adds or drops a row still holds
    the same table. Two tables on one page are always different products (refinance vs cash-out
    refinance can share row names), and a table joins the group whose product name fits best."""
    parent = {t: t for t in tables}
    pages = {t: {t[0]} for t in tables}

    def root(t):
        while parent[t] != t:
            t = parent[t]
        return t

    keys = sorted(tables)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rows = _jaccard(tables[a][0], tables[b][0])
            if rows >= SAME_TABLE:
                pairs.append((rows, _jaccard(tables[a][1], tables[b][1]), a, b))
    for _, _, a, b in sorted(pairs, reverse=True):
        ra, rb = root(a), root(b)
        if ra != rb and not pages[ra] & pages[rb]:
            parent[rb] = ra
            pages[ra] |= pages.pop(rb)
    return {t: root(t) for t in tables}


def _row_facts(store):
    """Group every %-valued cell of every row chunk by (snapshot, audience, table, tier, metric)."""
    rows, tables = [], defaultdict(dict)
    for rec in store.records:
        if rec.chunk_type != "row":
            continue
        for o in rec.occurrences:
            lines = [l for l in o.text.split("\n") if l.strip().startswith("|")]
            if len(lines) < 2:
                continue
            header, row = _cells(lines[0]), _cells(lines[-1])
            names = _row_names(store, o)
            ranks = _year_ranks(names)
            cols = tuple(_col_kind(c, v) for c, v in zip(header, row))
            family = (o.snapshot_id, _audience(o.url_norm), cols)
            label = o.label or (o.heading_path[-1] if o.heading_path else "")
            tables[family][(o.url_norm, o.table_id)] = (frozenset(_relabel(n, ranks) for n in names),
                                                        _label_words(label))
            rows.append((o, header, row, lines, ranks, family))

    same = {fam: _same_tables(ts) for fam, ts in tables.items()}
    groups = defaultdict(list)
    for o, header, row, lines, ranks, family in rows:
        table = same[family][(o.url_norm, o.table_id)]
        product = o.label or (o.heading_path[-1] if o.heading_path else o.title)
        tier = tuple(_relabel(_norm(v), ranks) for col, v in zip(header, row)
                     if _col_kind(col, v) == "tier" and not _SKIP_COL.search(col))
        eff = o.meta["dates"].get("effective")
        for col, v in zip(header, row):
            m = _PCT.search(v)
            if not m:
                continue
            metric = re.sub(r"\d+$", "", col).strip()
            groups[(family, table, tier, _metric(col, v))].append({
                "value": float(m.group(1)), "raw": _clean_value(v), "occ": o, "product": product,
                "row": row[0], "metric": metric, "line": lines[-1].strip(),
                "date": eff["date"] if eff else "",
            })
    return groups


def _winner(by_value: dict) -> float:
    def key(v):
        facts = by_value[v]
        return (max(f["date"] for f in facts), max(_TRUST.get(f["occ"].page_type, 0.5) for f in facts),
                len({f["occ"].url_norm for f in facts}), -v)
    return max(by_value, key=key)


def find_conflicts(store) -> list[Conflict]:
    # (stale table, stale page) -> {snapshot: [(winning facts, losing facts)]}
    themes = defaultdict(lambda: defaultdict(list))
    for ((sid, _aud, _cols), product, _tier, _metric), facts in _row_facts(store).items():
        # Name the topic by the table's label where any page gives one.
        labels = [f["occ"].label for f in facts if f["occ"].label]
        name = max(set(labels), key=labels.count) if labels else facts[0]["product"]
        by_value = defaultdict(list)
        for f in facts:
            f["product"] = name
            by_value[f["value"]].append(f)
        # A page with two values for one key is ambiguous, not a conflict: leave it out.
        urls_per_value = {v: {f["occ"].url_norm for f in fs} for v, fs in by_value.items()}
        ambiguous = {u for v, us in urls_per_value.items() for u in us
                     if any(u in us2 for v2, us2 in urls_per_value.items() if v2 != v)}
        by_value = {v: [f for f in fs if f["occ"].url_norm not in ambiguous] for v, fs in by_value.items()}
        by_value = {v: fs for v, fs in by_value.items() if fs}
        if len(by_value) < 2:
            continue
        win = _winner(by_value)
        for v, fs in by_value.items():
            if v == win:
                continue
            for url in sorted({f["occ"].url_norm for f in fs}):
                lose = [f for f in fs if f["occ"].url_norm == url][:1]
                themes[(lose[0]["occ"].table_id, url)][sid].append((by_value[win], lose))

    # One conflict per stale table, across crawls and mirror pages; the same stale values on
    # several pages count once.
    merged = {}
    for (_table, url), per_snap in themes.items():
        sig = frozenset((_norm(l["row"]), _norm(l["metric"]), l["value"]) for pairs in per_snap.values()
                        for _, lose in pairs for l in lose)
        merged.setdefault(sig, []).append((url, per_snap))

    out = []
    for entries in merged.values():
        stale_urls = sorted({u for u, _ in entries})
        snaps = sorted({s for _, ps in entries for s in ps})
        latest = snaps[-1]
        pairs = [p for _, ps in entries for p in ps.get(latest, [])]
        all_pairs = [p for _, ps in entries for s in snaps for p in ps.get(s, [])]
        win_facts = [w for ws, _ in pairs for w in ws]
        lose_facts = [l for _, ls in all_pairs for l in ls]
        win_main = max(win_facts, key=lambda f: (f["date"], _TRUST.get(f["occ"].page_type, 0.5)))
        lose_main = min(lose_facts, key=lambda f: f["date"] or "0")

        values = list(dict.fromkeys([f["raw"] for f in win_facts] + [f["raw"] for f in lose_facts]))
        rows = list(dict.fromkeys(f["row"] for f in win_facts))
        metrics = " / ".join(dict.fromkeys(f["metric"] for f in win_facts))
        label = win_main["product"]
        topic = (f"{label} ({', '.join(rows[:2])}{', …' if len(rows) > 2 else ''}) {metrics}: "
                 f"{', '.join(_path(u) for u in stale_urls)} vs {_path(win_main['occ'].url_norm)}")
        resolution = "; ".join(dict.fromkeys(f"{f['row']} {f['metric']} {f['raw']}" for f in win_facts))
        if win_main["date"] and win_main["date"] != lose_main["date"]:
            reason = (f"{_path(win_main['occ'].url_norm)} has the newer effective date ({win_main['date']}) than "
                      f"{_path(lose_main['occ'].url_norm)} ({lose_main['date'] or 'no date'}); the older table is a stale copy")
        else:
            reason = (f"same or no effective date; {_path(win_main['occ'].url_norm)} ({win_main['occ'].page_type} page) "
                      f"outranks {_path(lose_main['occ'].url_norm)} ({lose_main['occ'].page_type} page) for rates")

        sources = [Source(url=win_main["occ"].url, snapshot_id=win_main["occ"].snapshot_id, quote=win_main["line"][:200])]
        for url in stale_urls:
            f = max((l for l in lose_facts if l["occ"].url_norm == url), key=lambda l: l["occ"].snapshot_id)
            sources.append(Source(url=f["occ"].url, snapshot_id=f["occ"].snapshot_id, quote=f["line"][:200]))
        out.append(Conflict(topic=topic, values=values, sources=sources, resolution=resolution, reason=reason))
    out += _footer_conflicts(store)
    out.sort(key=lambda c: c.topic)
    return out


# Facts the "About Dupaco" footer under every news post states about the credit union itself.
# Member counts and assets are left out: they grow, so each post was true on its own date.
_FOOTER_FACTS = {
    "number of branch locations": re.compile(r"\b(\d+) branch(?:es)?(?: office)?(?: locations?)?\b"),
    "number of counties served": re.compile(r"\b(\d+) counties\b"),
}


def _footer_conflicts(store) -> list[Conflict]:
    """Footers in the newest crawl that disagree; the newest post's footer is current."""
    latest = max((o.snapshot_id for rec in store.records for o in rec.occurrences), default=None)
    seen = defaultdict(lambda: defaultdict(list))   # fact -> value -> [(date, url, sentence, occ)]
    for rec in store.records:
        if rec.chunk_type != "footer" or "serves residents" not in rec.text:
            continue
        for o in store.occurrences_in(rec, latest):
            for fact, pattern in _FOOTER_FACTS.items():
                for sentence in re.split(r"(?<=\.)\s+", rec.text):
                    m = pattern.search(sentence)
                    if m:
                        seen[fact][m.group(1)].append((_date(o), o.url, sentence.strip(), o))
    out = []
    for fact, by_value in seen.items():
        if len(by_value) < 2:
            continue
        newest = {v: max(hits, key=lambda h: h[:2]) for v, hits in by_value.items()}
        order = sorted(newest, key=lambda v: newest[v][:2], reverse=True)
        win = order[0]
        date, url, sentence, occ = newest[win]
        stale = ", ".join(f"{v} ({newest[v][0] or 'no date'})" for v in order[1:])
        sources = [Source(url=h[1], snapshot_id=h[3].snapshot_id, quote=h[2][:200]) for h in (newest[v] for v in order)]
        out.append(Conflict(
            topic=f"Dupaco {fact} (About Dupaco footer on news posts)",
            values=order, sources=sources, resolution=win,
            reason=f"the newest post ({date}, {_path(occ.url_norm)}) says {win}; older posts' footers still say {stale}"))
    return out
