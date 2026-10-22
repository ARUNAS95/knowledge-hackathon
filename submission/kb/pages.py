"""Per-page metadata: page type, post date, author, cleaned text, and page history across crawls."""
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from . import boilerplate
from .urls import is_junk_url, normalize_url

_MONTH_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s+(\d{4})\b")
_AUTHOR = re.compile(r"^\[([^\]]+)\]\([^)]*/author/[^)]*\)", re.M)

# Page type sets how much the resolver trusts a page. First match wins.
_TYPE_RULES = [
    ("listing",    re.compile(r"/(tag|author|category)/|[?&]category=|/newsroom$|/learn/blog$|/resources(/[a-z-]+)?$")),
    ("document",   re.compile(r"/hubfs/.*\.(pdf|xlsx?|docx?)$", re.I)),
    ("news",       re.compile(r"/newsroom/")),
    ("blog",       re.compile(r"/learn/blog/|/\d{4}/\d{2}/\d{2}/|/business/resources/")),
    ("calculator", re.compile(r"/calculators?/|-calculators?$")),
    ("rates",      re.compile(r"-rates$|/rates$")),
    ("contact",    re.compile(r"/contact$|/locations")),
    ("product",    re.compile(r"/(personal|business)/|/switch-and-earn|/connect/|/first-time-homebuyer")),
    ("about",      re.compile(r"/about|/community|/careers")),
]


def page_type(url_norm: str) -> str:
    path = url_norm.split("dupaco.com", 1)[-1]
    if path in ("", "/"):
        return "home"
    for name, rx in _TYPE_RULES:
        if rx.search(path):
            return name
    return "page"


def _iso(m: re.Match) -> Optional[str]:
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date().isoformat()
    except ValueError:
        return None


@dataclass
class PageRecord:
    snapshot_id: str
    taken_on: str
    fetched_at: str
    url: str
    url_norm: str
    title: str
    page_type: str
    type_method: str
    post_date: Optional[str]
    author: Optional[str]
    clean: str
    clean_hash: str
    raw: str


def build_page(page, snapshot) -> Optional[PageRecord]:
    if is_junk_url(page.url):
        return None
    url_norm = normalize_url(page.url)
    header, body = boilerplate.split_header(page.markdown)
    post_date = author = None
    if header:
        d = _MONTH_DATE.search(header)
        post_date = _iso(d) if d else None
        a = _AUTHOR.search(header)
        author = a.group(1) if a else None
    clean = boilerplate.clean_markdown(body)
    return PageRecord(
        snapshot_id=snapshot.snapshot_id, taken_on=snapshot.taken_on, fetched_at=page.fetched_at,
        url=page.url, url_norm=url_norm, title=re.sub(r"\s*\|\s*Dupaco\s*$", "", page.title or ""),
        page_type=page_type(url_norm), type_method="url_rule",
        post_date=post_date, author=author, clean=clean,
        clean_hash=hashlib.sha1(clean.encode()).hexdigest()[:16], raw=page.markdown,
    )


class PageHistory:
    """url_norm -> {snapshot_id: clean_hash}, to see when a page appeared, changed or vanished."""

    def __init__(self):
        self.by_url: dict[str, dict[str, str]] = {}
        self.snapshots: list[str] = []

    def record(self, snapshot_id: str, pages: list[PageRecord]) -> dict:
        self.snapshots.append(snapshot_id)
        prev = self.snapshots[-2] if len(self.snapshots) > 1 else None
        seen = set()
        diff = {"added": 0, "changed": 0, "same": 0, "removed": 0}
        for p in pages:
            seen.add(p.url_norm)
            h = self.by_url.setdefault(p.url_norm, {})
            if prev is None or prev not in h:
                diff["added"] += 1
            elif h[prev] != p.clean_hash:
                diff["changed"] += 1
            else:
                diff["same"] += 1
            h[snapshot_id] = p.clean_hash
        if prev:
            diff["removed"] = sum(1 for u, h in self.by_url.items() if prev in h and u not in seen)
        return diff

    def present_in(self, url_norm: str) -> list[str]:
        return list(self.by_url.get(url_norm, {}))
