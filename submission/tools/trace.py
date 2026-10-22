#!/usr/bin/env python3
"""Show every step of answering one question. Also saved to results/trace_<id>.md.

    .venv/bin/python submission/tools/trace.py --id dup-087
    .venv/bin/python submission/tools/trace.py "What's the 17-month certificate rate?" [--as-of 2026-05-15]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402
from scoring import score_item  # noqa: E402
import solution  # noqa: E402
from config import MIN_BM25, MIN_COSINE, RRF_K, SEARCH_TOP_K  # noqa: E402
from kb import generate  # noqa: E402
from kb.resolver import LIST_QUESTION, LINK_QUESTION, topic_of  # noqa: E402

OUT: list[str] = []


def p(line: str = ""):
    print(line)
    OUT.append(line)


def path(url: str) -> str:
    return url.split("dupaco.com", 1)[-1] or "/"


def snip(text: str, n: int = 90) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def meta_lines(occ, indent="      "):
    m = occ.meta
    d = m["dates"]
    dates = {k: v["date"] for k, v in (("effective", d.get("effective")), ("post", d.get("post")),
                                         ("expiration", d.get("expiration"))) if v}
    qual = {k: v for k, v in m["qualifiers"].items() if v not in (False, None, [], {})}
    p(f"{indent}page: {occ.title!r} ({occ.page_type} page)  crawl {occ.snapshot_id}  url {occ.url}")
    p(f"{indent}chunk: {occ.chunk_type}  heading {' > '.join(occ.heading_path) or '-'}  label {occ.label or '-'}")
    p(f"{indent}dates {dates or '-'}  tier {qual or '-'}  warnings {m['warnings'] or '-'}")
    if m.get("links"):
        p(f"{indent}links {[(l['text'], l['url']) for l in m['links'][:4]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--id", help="eval question id, e.g. dup-087")
    ap.add_argument("--as-of")
    ap.add_argument("--eval", default=str(ROOT / "eval/public_eval.jsonl"))
    ap.add_argument("-n", type=int, default=12, help="rows to show per search list")
    args = ap.parse_args()

    item = None
    if args.id:
        item = next(json.loads(l) for l in open(args.eval) if json.loads(l)["id"] == args.id)
        question, as_of = item["question"], item.get("as_of")
    else:
        question, as_of = args.question, args.as_of
    if not question:
        ap.error("give a question or --id")

    ks = solution.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)

    # Wrap the functions answer() calls to record each step.
    rec_state: dict = {}
    real_resolve, real_write, real_validate = solution.resolve, solution.write_answer, solution.validate

    def resolve(*a, **k):
        k["trace"] = rec_state.setdefault("resolver", {})
        out = real_resolve(*a, **k)
        rec_state["evidence"] = out
        return out

    def write_answer(llm, q, items, taken_on=""):
        rec_state["prompt"] = (generate.system_prompt(), generate.user_message(q, items, taken_on))
        out = real_write(llm, q, items, taken_on)
        rec_state["llm"] = out
        return out

    def validate(*a, **k):
        out = real_validate(*a, **k)
        rec_state["validator"] = out
        return out

    solution.resolve, solution.write_answer, solution.validate = resolve, write_answer, validate

    sid = ks.select_snapshot(as_of)
    p(f"# Trace: {question}")
    p(f"as_of: {as_of or 'none (answer from the newest crawl)'}")
    p(f"crawls: {', '.join(s for _, s in ks.snapshots)}")
    p(f"-> crawl used: {sid}  (latest crawl taken on or before as_of)")
    if item:
        p(f"eval {item['id']} [{item['category']}, weight {item.get('weight', 1)}]: must_include {item.get('must_include')} "
          f"must_not_include {item.get('must_not_include')} expected_sources {item.get('expected_sources')}")

    p("\n## 2. How the question is read (rules, no LLM: USE_INTENT_LLM is off)")
    p(f"topic: {topic_of(question)}   list question: {bool(LIST_QUESTION.search(question))}   "
      f"link question: {bool(LINK_QUESTION.search(question))}")

    recs = ks.store.records
    qv = ks.embedder.embed_query(question)
    kw = ks.index.search(question, allowed=lambda cid: sid in recs[cid].visible_in, k=SEARCH_TOP_K)
    vec = ks.vectors.search(qv, sid, SEARCH_TOP_K)
    total = sum(1 for r in recs if sid in r.visible_in)
    p(f"\n## 3. Search (both lists only see chunks visible in {sid}: {total} of {len(recs)} unique chunks)")
    p(f"### 3a. Keyword (BM25) top {args.n} of {len(kw)}")
    for i, (cid, sc) in enumerate(kw[: args.n]):
        o = ks.store.occurrences_in(recs[cid], sid)[0]
        p(f"  {i + 1:2}. bm25 {sc:5.1f}  #{cid:<6} {o.chunk_type:7} {path(o.url_norm)[:50]:50} | {snip(recs[cid].text, 60)}")
    p(f"### 3b. Vector (bge-small, cosine) top {args.n} of {len(vec)}")
    for i, (cid, cos) in enumerate(vec[: args.n]):
        o = ks.store.occurrences_in(recs[cid], sid)[0]
        p(f"  {i + 1:2}. cos {cos:.3f}  #{cid:<6} {o.chunk_type:7} {path(o.url_norm)[:50]:50} | {snip(recs[cid].text, 60)}")

    fused: dict[int, float] = {}
    kw_rank = {cid: r for r, (cid, _) in enumerate(kw)}
    vec_rank = {cid: r for r, (cid, _) in enumerate(vec)}
    for lst in (kw, vec):
        for rank, (cid, _) in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    bm = dict(kw)
    dropped = [cid for cid in fused if float(ks.matrix[cid] @ qv) < MIN_COSINE and bm.get(cid, 0) < MIN_BM25]
    hits = ks.search(question, sid, qv=qv)
    p(f"\n## 4. Merged (reciprocal-rank fusion) -> score filter (keep if cos >= {MIN_COSINE} or bm25 >= {MIN_BM25})")
    p(f"{len(fused)} merged, {len(dropped)} dropped by the filter, {len(hits)} kept. Relevance = 0.5 x scaled cos + 0.5 x scaled bm25.")
    for i, (r, rel, cos, b) in enumerate(hits[: args.n]):
        o = ks.store.occurrences_in(r, sid)[0]
        kr = kw_rank.get(r.content_id)
        vr = vec_rank.get(r.content_id)
        p(f"  {i + 1:2}. rel {rel:.2f} (cos {cos:.2f}, bm25 {b:4.1f}; kw #{kr + 1 if kr is not None else '-'}, "
          f"vec #{vr + 1 if vr is not None else '-'})  #{r.content_id:<6} {o.chunk_type:7} {path(o.url_norm)[:45]} | {snip(r.text, 50)}")
        meta_lines(o, "        ")

    ans = ks._answer(question, as_of)
    tr = rec_state.get("resolver", {})

    p("\n## 5. Resolver")
    if tr:
        p(f"topic {tr['topic']}  tiers asked {tr['tiers'] or '-'} amounts {tr['amounts'] or '-'}  max items {tr['max_items']}")
        p(f"trust table: {tr['trust_table']}")
        p(f"link words {tr['link_words'] or '-'}  link pages {[path(u) for u in tr['link_pages']] if tr['link_words'] else '-'}  "
          f"digest page {path(tr['digest_page']) if tr['digest_page'] else '-'}  membership question {tr['membership_question']}")
        p("### decisions")
        for s in tr.get("steps", []):
            extra = {k: v for k, v in s.items() if k not in ("stage", "content_id", "type", "url", "why")}
            p(f"  {s['stage']:8} #{s['content_id']:<6} {s['type']:7} {path(s['url'])[:45]:45} {s['why']} {extra or ''}")
        p("### ranked candidates (score = relevance x (0.7 + 0.3 x trust); tier match first, then score, then date)")
        for i, e in enumerate(tr.get("ranked", [])[:15]):
            p(f"  {i + 1:2}. score {e['score']:.3f} (rel {e['relevance']:.2f}, trust {e['trust']:.2f}, tier {e['tier']}, date {e['date'] or '-'})"
              f"  #{e['content_id']:<6} {e['type']:7} {path(e['url'])[:45]} | {snip(e['text'], 50)}")
    else:
        p("(resolver not reached)")

    ev = rec_state.get("evidence") or []
    p(f"\n## 6. Evidence given to the writer ({len(ev)} items)")
    for i, e in enumerate(ev, 1):
        p(f"  [{i}] #{e.rec.content_id} score {e.relevance * (0.7 + 0.3 * e.trust):.3f}  urls {[path(u) for u in e.urls[:3]]}")
        meta_lines(e.occ)
        p(f"      reasons: {e.reasons}")

    p("\n## 7. Writer (LLM) and checks")
    if "prompt" in rec_state:
        p(f"model {ks.llm.model}; system prompt = kb/generate.py SYSTEM ({len(rec_state['prompt'][0])} chars)")
        p("### exact user message sent")
        for line in rec_state["prompt"][1].split("\n"):
            p("    " + line)
        out = rec_state.get("llm")
        p(f"### raw reply: {json.dumps(out, ensure_ascii=False) if out else '(LLM failed -> template)'}")
        if "validator" in rec_state:
            text, drops = rec_state["validator"]
            p(f"### validator (every number/URL must be in the evidence): dropped {drops or 'nothing'}")
    else:
        p("no LLM call (no evidence, no key, or reduced mode)")
    if ans.text.startswith("I don't know"):
        p("refusal: writer found=false, or a 'what rate/fee' answer without a value (solution.states_a_value)")

    p("\n## 8. Final answer")
    p(f"confident {ans.confident}\n{ans.text}")
    for s in ans.sources:
        p(f"  source {s.url}  ({s.snapshot_id})  quote: {snip(s.quote or '', 100)!r}")
    if item:
        sc = score_item(item, ans)
        p(f"\nscore {sc['score']} (correctness {sc['correctness']}, source hit {sc['source_hit']})  "
          f"matched {sc['matched']} missed {sc['missed']} traps {sc['trap_hits']}")

    name = args.id or re.sub(r"[^a-z0-9]+", "-", question.lower())[:40]
    out_path = ROOT / "results" / f"trace_{name}.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(OUT))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
