"""Dupaco knowledge system.

Ingest (no LLM): clean pages, chunk, embed, and build a keyword index.
Answer: pick the crawl -> hybrid search -> resolver -> LLM writer -> validator -> Answer.
Without the LLM it answers from a template or refuses.
"""
import concurrent.futures as cf
import hashlib
import json
import logging
import re
import sys
import threading
from urllib.parse import urlsplit
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

import numpy as np  # noqa: E402

from config import (DEV_ANSWER_CACHE, EVIDENCE_ITEMS, MAX_EMBED_TOKENS, MIN_BM25, MIN_COSINE, RRF_K,  # noqa: E402
                    SEARCH_TOP_K, USE_INTENT_LLM)
from kb import templates  # noqa: E402
from kb.chunker import chunk_page, enforce_token_limit  # noqa: E402
from kb.conflicts import find_conflicts  # noqa: E402
from kb.embeddings import Embedder  # noqa: E402
from kb.generate import write_answer  # noqa: E402
from kb.llm import LLMClient, LLMError  # noqa: E402
from kb.metadata import occurrence_metadata  # noqa: E402
from kb.pages import PageHistory, build_page  # noqa: E402
from kb.parse import parse_intent  # noqa: E402
from kb.resolver import resolve  # noqa: E402
from kb.retrieval import BM25Index, tokenize  # noqa: E402
from kb.store import ChunkStore, Occurrence  # noqa: E402
from kb.urls import normalize_url  # noqa: E402
from kb.validator import _URL, numbers, validate  # noqa: E402
from kb.vector_store import VectorStore  # noqa: E402

log = logging.getLogger("dupaco")

_CAP = re.compile(r"\b(?:up to|within|by|no more than|more than|at most|exceed)\s+\d+(?:\.\d+)?\s?%", re.I)


# Percentages that are not a price, e.g. "a 20% down payment".
_NOT_RATE = re.compile(r"\d+(?:\.\d+)?\s?%\s+(?:[a-z-]+\s+)?(?:down|of (?:the )?(?:construction |purchase |home |appraised )?"
                       r"(?:costs?|price|value)|loan[- ]to[- ]value|ltv|financing)", re.I)


def states_a_value(answer: str) -> bool:
    """True if the answer states an actual rate or amount, not just a cap or down payment."""
    return bool(re.search(r"\d(?:[\d,.]*)\s*%|\$\s?\d", _NOT_RATE.sub("", _CAP.sub("", answer))))


ASKS_FOR_LINK = re.compile(r"\b(link|url|website|web page|webpage|where can i (?:see|find))\b", re.I)
ASKS_FOR_VALUE = re.compile(
    r"\b(what|how much)\b[^?]*\b(rate|apr|apy|interest|dividend|fee|cost|charge|price)s?\b", re.I)


HOURS_QUESTION = re.compile(r"\b(hours?|lobby|drive[- ]?(?:up|thru|through))\b", re.I)


_SLUG_SKIP ={"https", "www", "dupaco", "com", "hubfs", "learn", "blog", "newsroom", "pdf", "personal"}


def context_title(page) -> str:
    """Page title plus the words of its URL path (skipped for articles, whose slug repeats the title)."""
    if page.page_type in ("blog", "news", "listing"):
        return page.title
    words = [w for w in re.split(r"[^a-z0-9]+", page.url_norm.lower().split("?")[0]) if w and w not in _SLUG_SKIP]
    return f"{page.title} ({' '.join(words)})" if words else page.title


class DupacoKnowledge:
    def __init__(self):
        self.store = ChunkStore()
        self.history = PageHistory()
        self.index = BM25Index()
        self.embedder = Embedder()
        self.vectors = VectorStore()
        self.matrix = np.zeros((0, self.embedder.dim), np.float32)  # row i = record i's vector
        self.snapshots: list[tuple[str, str]] = []      # (taken_on, snapshot_id), oldest first
        self.pages: dict[str, dict] = {}                # snapshot_id -> url_norm -> PageRecord
        self.ingest_log: list[dict] = []
        self.llm = LLMClient()
        self.pool = cf.ThreadPoolExecutor(max_workers=2)
        self._cache: dict[tuple, Answer] = {}           # same question twice -> same answer
        self._lock = threading.Lock()
        self._dev_cache = DevAnswerCache(self.llm.model) if DEV_ANSWER_CACHE else None

    # ---- ingest ------------------------------------------------------------------------------
    def ingest(self, snapshot: Snapshot) -> None:
        sid = snapshot.snapshot_id
        pages = {}
        for p in snapshot.pages:
            rec = build_page(p, snapshot)
            if rec is None:
                continue
            # Two URLs can normalise to one page: keep the most recently fetched.
            old = pages.get(rec.url_norm)
            if old is None or rec.fetched_at > old.fetched_at:
                pages[rec.url_norm] = rec
        diff = self.history.record(sid, list(pages.values()))
        n_chunks = 0
        count = self.embedder.count_tokens
        for page in pages.values():
            title = context_title(page)
            chunks = enforce_token_limit(chunk_page(page.clean, page.url_norm), title, count, MAX_EMBED_TOKENS)
            for c in chunks:
                occ = Occurrence(
                    snapshot_id=sid, taken_on=page.taken_on, fetched_at=page.fetched_at,
                    url=page.url, url_norm=page.url_norm, title=page.title, page_type=page.page_type,
                    chunk_index=c.chunk_index, chunk_type=c.chunk_type, heading_path=c.heading_path,
                    label=c.label, table_id=c.table_id, row_index=c.row_index, text=c.text,
                    meta=occurrence_metadata(c, page),
                )
                self.store.add(c.embed_text(title), occ)
                n_chunks += 1
        self.pages[sid] = pages
        self.snapshots.append((snapshot.taken_on, sid))
        self.snapshots.sort()
        carried = self._carry_forward(sid, pages)

        recs = self.store.records
        self.matrix = self.embedder.embed_documents([r.embed_text for r in recs])
        self.vectors.rebuild([r.content_id for r in recs], self.matrix, [sorted(r.visible_in) for r in recs])
        self.index.build(recs)
        self.ingest_log.append({"snapshot": sid, "pages": len(pages), "chunks": n_chunks,
                                "unique_total": len(recs), "carried_pages": carried, **diff})

    def _carry_forward(self, sid: str, pages: dict) -> int:
        """Keep pages missing from this crawl visible from their latest earlier copy."""
        missing = {}
        for url, seen in self.history.by_url.items():
            if url not in pages:
                earlier = [s for s in seen if s < sid]
                if earlier:
                    missing[url] = max(earlier)
        if not missing:
            return 0
        for rec in self.store.records:
            if any(o.url_norm in missing and o.snapshot_id == missing[o.url_norm] for o in rec.occurrences):
                rec.carried_into.add(sid)
        return len(missing)

    # ---- search ------------------------------------------------------------------------------
    def select_snapshot(self, as_of=None) -> str:
        """Latest crawl taken on or before as_of (newest crawl when as_of is None)."""
        if not as_of:
            return self.snapshots[-1][1]
        eligible = [sid for taken, sid in self.snapshots if taken <= as_of[:10]]
        return eligible[-1] if eligible else self.snapshots[0][1]

    def search(self, question: str, snapshot_id: str, k: int = SEARCH_TOP_K, filtered: bool = True,
               extra_terms: str = "", qv=None):
        """Hybrid search in one crawl -> [(record, relevance 0..1, cosine, bm25)], best first."""
        recs = self.store.records
        kw = self.index.search(f"{question} {extra_terms}", allowed=lambda cid: snapshot_id in recs[cid].visible_in, k=k)
        if qv is None:
            qv = self.embedder.embed_query(question)
        vec = self.vectors.search(qv, snapshot_id, k)
        fused: dict[int, float] = {}
        for lst in (kw, vec):
            for rank, (cid, _) in enumerate(lst):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        bm25 = dict(kw)
        rows = []
        for cid, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0])):
            cos = float(self.matrix[cid] @ qv)
            b = bm25.get(cid, 0.0)
            if filtered and cos < MIN_COSINE and b < MIN_BM25:
                continue
            rows.append((cid, cos, b))
        if not rows:
            return []
        # Fused rank scores are too flat to compare, so relevance mixes the scaled raw scores.
        cmin, cmax = min(r[1] for r in rows), max(r[1] for r in rows)
        bmax = max(r[2] for r in rows) or 1.0
        out = []
        for cid, cos, b in rows:
            rel = 0.5 * ((cos - cmin) / (cmax - cmin) if cmax > cmin else 1.0) + 0.5 * b / bmax
            out.append((recs[cid], rel, cos, b))
        out.sort(key=lambda x: (-x[1], x[0].content_id))
        return out

    # ---- answer ------------------------------------------------------------------------------
    def answer(self, question, as_of=None):
        key = (re.sub(r"\s+", " ", question.strip().lower()), (as_of or "")[:10])
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        ans = self._dev_cache.get(key) if self._dev_cache else None
        if ans is None:
            ans = self._answer(question, as_of)
            if self._dev_cache:
                self._dev_cache.put(key, ans)
        with self._lock:
            self._cache[key] = ans
        return ans

    def _answer(self, question, as_of):
        sid = self.select_snapshot(as_of)
        taken_on = dict((s, t) for t, s in self.snapshots)[sid]
        # Intent parsing, when enabled, runs while the question is embedded.
        f_intent = self.pool.submit(parse_intent, self.llm, question) if USE_INTENT_LLM else None
        qv = self.embedder.embed_query(question)
        intent = f_intent.result() if f_intent else {}
        hits = self.search(question, sid, extra_terms=intent.get("search_query", ""), qv=qv)
        evidence = resolve(question, [(r, rel) for r, rel, _, _ in hits], self.store, sid, max_items=EVIDENCE_ITEMS, intent=intent)
        if not evidence:
            return Answer(text=templates.REFUSAL, confident=False)

        prefix = templates.as_of_prefix(taken_on) if as_of else ""
        system_text = prefix
        used = evidence[:1]
        if self.llm.available:
            try:
                out = write_answer(self.llm, question, evidence, taken_on)
            except LLMError as e:
                log.warning("writer failed, using template: %s", e)
                out = None
            if out is not None and not out["found"]:
                return Answer(text=templates.REFUSAL, confident=False)
            if (out is not None and ASKS_FOR_VALUE.search(question) and not ASKS_FOR_LINK.search(question)
                    and not states_a_value(out["answer"])):
                # A rate question answered without a rate means the site doesn't list it.
                log.info("value question answered without a value -> refusal: %s", out["answer"][:120])
                return Answer(text=templates.REFUSAL, confident=False)
            if out is not None:
                used = [evidence[i - 1] for i in out["used_ids"]] or evidence[:1]
                # Page URLs count as evidence too, since the writer sees them.
                text, dropped = validate(out["answer"], [e.text for e in evidence] + [u for e in evidence for u in e.urls],
                                         question, system_text)
                if dropped:
                    log.info("validator dropped %d sentence(s): %s", len(dropped), dropped)
                if not text:
                    text = templates.from_evidence(evidence[0])
                    used = evidence[:1]
            else:
                text = templates.from_evidence(evidence[0])
        else:
            text = templates.from_evidence(evidence[0])

        if prefix:
            first = text.split(" ", 1)[0]
            text = prefix + (text[0].lower() + text[1:] if first in ("The", "A", "An", "Your", "It", "This", "As", "On", "In", "Back", "At", "Yes,", "No,") else text)
        sources = self._sources(used, text, sid, question)
        if HOURS_QUESTION.search(question):
            sources = self._with_locator(sources, sid)
        return Answer(text=text, sources=sources)

    def _with_locator(self, sources: list, sid: str) -> list:
        """Also cite the branch locator, since hours differ by branch."""
        page = self.pages.get(sid, {}).get("https://www.dupaco.com/contact")
        if page is None or any(s.url.rstrip("/").endswith("/contact") for s in sources):
            return sources
        lines = [l for l in page.clean.split("\n") if l.strip()]
        quote = next((l for l in lines if "branch or ATM" in l), lines[0] if lines else "")
        return sources[:4] + [Source(url=page.url, snapshot_id=sid, quote=quote.strip()[:200])]

    def _sources(self, used, text, sid, question="") -> list:
        """Cite the pages of the evidence the writer used, each with a verbatim quote.

        The quote is the evidence line holding the answer's first value, then the best match on
        numbers, links and words. Pages of the asked audience (personal/business) whose URL fits
        the topic come first."""
        plain = _URL.sub("", text)
        first = _FIRST_VALUE.search(plain)
        first_nums = numbers(first.group(0)) if first else set()
        ans_nums = numbers(plain)
        wants_business = bool(re.search(r"\bbusiness", question, re.I))
        q_words = set(tokenize(question))
        ans_urls = {_link_key(u) for u in _URL.findall(text)}
        ans_words = set(tokenize(text))
        pages = self.pages.get(sid, {})
        out, seen, quotes = [], set(), []

        def add(url, snapshot_id, quote):
            if url not in seen:
                seen.add(url)
                out.append(Source(url=url, snapshot_id=snapshot_id, quote=quote.strip()[:200]))

        def best(lines):
            return max(lines, key=lambda l: (bool(first_nums & numbers(_URL.sub("", l))),
                                             len(numbers(_URL.sub("", l)) & ans_nums),
                                             any(_link_key(u) in ans_urls for u in _URL.findall(l)),
                                             len(set(tokenize(l)) & ans_words)), default="")

        for e in used:
            lines = [l.strip() for l in e.rec.text.split("\n") if l.strip() and not l.startswith("## ")]
            quote = best(lines)
            quotes.append(quote)
            topic = set(tokenize(" ".join([e.occ.label, e.occ.title] + e.occ.heading_path)))

            def own_quote(url):
                """The quote taken from this page's own lines (combined items span pages)."""
                page = pages.get(normalize_url(url))
                if page is None or quote in page.clean:
                    return quote
                return best([l for l in lines if l in page.clean]) or quote

            def fit(url):
                page = pages.get(normalize_url(url))
                q = own_quote(url)
                score = len(numbers(_URL.sub("", q)) & ans_nums)
                words = set(tokenize(re.sub(r"[^a-z0-9]+", " ", urlsplit(url).path.lower())))
                return (_is_business(url) != wants_business, page is None or q not in page.clean, -score,
                        -(len(words & topic) - 0.5 * len(words - topic)))
            for url in sorted(e.urls, key=fit)[:3]:
                add(url, e.occ.snapshot_id, own_quote(url))
        q = quotes[0].strip() if quotes else ""
        if len(q) >= 25 and len(out) < 5:
            wrong_audience = all(_is_business(src.url) != wants_business for src in out)
            candidates = []
            for page in pages.values():
                if q not in page.clean:
                    continue
                path_words = set(tokenize(re.sub(r"[^a-z0-9]+", " ", urlsplit(page.url).path.lower())))
                # Another page repeating the quote is worth citing if it is about the topic, or if
                # it is the asked audience's copy and every cited page so far is the other one.
                if not path_words or path_words & (q_words | ans_words) or \
                        (wrong_audience and _is_business(page.url) == wants_business):
                    candidates.append((len(path_words - q_words - ans_words), page.url))
            for _, url in sorted(candidates)[:min(2, 5 - len(out))]:
                add(url, sid, q)
        out.sort(key=lambda src: _is_business(src.url) != wants_business)
        return out[:5]

    def conflicts(self):
        return find_conflicts(self.store)


_FIRST_VALUE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%")


def _is_business(url: str) -> bool:
    return urlsplit(url).path.startswith("/business")


def _link_key(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.lower()).rstrip("/.")


class DevAnswerCache:
    """Development only: answers saved across runs, keyed by question and a hash of code and model."""

    def __init__(self, model: str):
        root = Path(__file__).resolve().parent
        h = hashlib.sha1(model.encode())
        for f in sorted(root.rglob("*.py")):
            if ".cache" not in f.parts and "tools" not in f.parts:
                h.update(f.read_bytes())
        self.path = root / ".cache" / "dev_answers" / f"{h.hexdigest()[:16]}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.lock = threading.Lock()

    def get(self, key):
        d = self.data.get(json.dumps(key))
        if d is None:
            return None
        return Answer(text=d["text"], confident=d["confident"], conflict_note=d.get("conflict_note"),
                      sources=[Source(**s) for s in d["sources"]])

    def put(self, key, ans: Answer):
        with self.lock:
            self.data[json.dumps(key)] = {"text": ans.text, "confident": ans.confident,
                                          "conflict_note": ans.conflict_note,
                                          "sources": [vars(s) for s in ans.sources]}
            self.path.write_text(json.dumps(self.data))


def build():
    return DupacoKnowledge()
