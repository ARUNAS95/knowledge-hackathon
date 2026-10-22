#!/usr/bin/env python3
"""Show, for each eval question, the first stage (search, resolver, writer, final) that loses
an expected value. Writes results/stage_report.html and results/stage_report.json.

    .venv/bin/python submission/tools/stage_report.py [--eval eval/public_eval.jsonl]
"""
import argparse
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402
from scoring import REFUSAL_RE, contains, score_item  # noqa: E402
import solution  # noqa: E402

STAGES = ["search", "resolver", "writer", "final"]
STAGE_HELP = {
    "search": "in any of the ~40 chunks search returned for the crawl",
    "resolver": "in the evidence items the resolver gave the writer",
    "writer": "in the LLM's raw reply",
    "final": "in the final answer (after validator and refusal checks)",
}
# Optional notes per question id, shown next to the verdict.
KNOWN = {}


def path(url: str) -> str:
    return url.split("dupaco.com", 1)[-1] or "/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "eval/public_eval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.eval)]

    ks = solution.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)

    cap: dict = {}
    real_resolve, real_write, real_validate = solution.resolve, solution.write_answer, solution.validate

    def resolve(*a, **k):
        cap["evidence"] = real_resolve(*a, **k)
        return cap["evidence"]

    def write_answer(*a, **k):
        cap["raw"] = real_write(*a, **k)
        return cap["raw"]

    solution.resolve, solution.write_answer = resolve, write_answer
    rows = []
    for it in items:
        cap.clear()
        q, as_of = it["question"], it.get("as_of")
        sid = ks.select_snapshot(as_of)
        hits = ks.search(q, sid)
        t0 = time.perf_counter()
        ans = ks._answer(q, as_of)
        latency = time.perf_counter() - t0
        ev = cap.get("evidence") or []
        raw = (cap.get("raw") or {}).get("answer", "") if cap.get("raw") else ""
        texts = {
            "search": "\n".join(r.text for r, *_ in hits) + "\n" + " ".join(o.url for r, *_ in hits for o in ks.store.occurrences_in(r, sid)),
            "resolver": "\n".join(e.text for e in ev) + "\n" + " ".join(u for e in ev for u in e.urls),
            "writer": raw,
            "final": ans.text,
        }
        must, traps = it.get("must_include", []), it.get("must_not_include", [])
        found = {st: [t for t in must if contains(texts[st], t)] for st in STAGES}
        trap_in = {st: [t for t in traps if contains(texts[st], t)] for st in STAGES}
        sc = score_item(it, ans)
        refused = (not ans.confident) or bool(REFUSAL_RE.search(ans.text))
        if it["category"] == "should_refuse":
            verdict = "ok (refused)" if sc["correctness"] == 1 else "answered instead of refusing"
            lost = "-" if sc["correctness"] == 1 else "writer"
        elif not must:
            verdict, lost = "no must_include (needs --judge)", "-"
        else:
            lost = next((st for st in STAGES if len(found[st]) < len(must)), None)
            if lost is None:
                verdict = "ok" if not sc["trap_hits"] else "stale value in final answer"
                lost = "-" if not sc["trap_hits"] else "final"
            else:
                gone = [t for t in must if t not in found[lost]]
                verdict = f"lost at {lost}: {gone}"
        rows.append({
            "id": it["id"], "category": it["category"], "weight": it.get("weight", 1), "question": q,
            "as_of": as_of, "crawl": sid, "must": must, "traps": traps, "found": found, "trap_in": trap_in,
            "latency": round(latency, 2), "score": sc["score"], "source_hit": sc["source_hit"], "verdict": verdict, "lost": lost,
            "refused": refused, "known": KNOWN.get(it["id"], ""),
            "search_top": [(round(rel, 2), r.chunk_type, path(ks.store.occurrences_in(r, sid)[0].url_norm), r.text[:90])
                           for r, rel, *_ in hits[:5]],
            "evidence": [(e.occ.chunk_type, [path(u) for u in e.urls[:2]], e.text[:160]) for e in ev],
            "raw": raw, "final": ans.text, "sources": [path(s.url) for s in ans.sources],
        })
        print(f"{it['id']:8} {sc['score']:.2f}  {rows[-1]['verdict']}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_report.json").write_text(json.dumps(rows, indent=1))
    write_html(rows, out / "stage_report.html", args.eval)
    summary(rows)
    print(f"\nsaved {out / 'stage_report.html'}  (open it in a browser)")


def summary(rows):
    print("\n=== Funnel: share of expected values still present after each stage ===")
    tot = sum(len(r["must"]) for r in rows)
    for st in STAGES:
        n = sum(len(r["found"][st]) for r in rows)
        print(f"  {st:9} {n:3}/{tot}  ({n / tot:.0%})   {STAGE_HELP[st]}")
    print("\n=== Stale/wrong values reaching each stage (questions) ===")
    for st in STAGES:
        print(f"  {st:9} {sum(1 for r in rows if r['trap_in'][st])}")
    print("\n=== Where questions are lost ===")
    for k, v in Counter(r["lost"] for r in rows).most_common():
        print(f"  {k:9} {v}")


def write_html(rows, dest: Path, eval_name: str):
    def cell(r, st):
        must = r["must"]
        if not must:
            return '<td class="na">–</td>'
        n = len(r["found"][st])
        cls = "ok" if n == len(must) else ("part" if n else "bad")
        trap = " ⚠" if r["trap_in"][st] else ""
        title = html.escape(f"found {r['found'][st]}; stale/wrong present: {r['trap_in'][st]}")
        return f'<td class="{cls}" title="{title}">{n}/{len(must)}{trap}</td>'

    tot = sum(len(r["must"]) for r in rows)
    funnel = "".join(f"<li><b>{st}</b>: {sum(len(r['found'][st]) for r in rows)}/{tot} — {STAGE_HELP[st]}</li>" for st in STAGES)
    body = []
    for r in rows:
        details = (
            f"<p><b>Crawl:</b> {r['crawl']} {('(as_of ' + r['as_of'] + ')') if r['as_of'] else ''} &nbsp; "
            f"<b>Needs:</b> {html.escape(str(r['must']))} &nbsp; <b>Must not say:</b> {html.escape(str(r['traps']))}</p>"
            "<p><b>Search top 5</b> (relevance, type, page, text):</p><ol>"
            + "".join(f"<li>{rel} {t} <code>{html.escape(p)}</code> {html.escape(x)}</li>" for rel, t, p, x in r["search_top"])
            + "</ol><p><b>Evidence given to the writer:</b></p><ol>"
            + "".join(f"<li>{t} <code>{html.escape(', '.join(u))}</code> {html.escape(x)}…</li>" for t, u, x in r["evidence"])
            + f"</ol><p><b>Writer raw reply:</b> {html.escape(r['raw'])}</p>"
            f"<p><b>Final answer:</b> {html.escape(r['final'])}</p>"
            f"<p><b>Sources:</b> {html.escape(', '.join(r['sources']))} &nbsp; <b>source hit:</b> {r['source_hit']}</p>"
        )
        body.append(
            f"<tr><td><b>{r['id']}</b><br><small>{r['category']} · w{r['weight']}</small></td>"
            f"<td>{html.escape(r['question'])}<details><summary>details</summary>{details}</details></td>"
            + "".join(cell(r, st) for st in STAGES)
            + f"<td class=\"{'ok' if r['score'] == 1 else ('part' if r['score'] else 'bad')}\">{r['score']:.2f}</td>"
            f"<td>{html.escape(r['verdict'])}{('<br><small>' + html.escape(r['known']) + '</small>') if r['known'] else ''}</td></tr>")
    dest.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage Report</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--ok:#d8f3dc;--part:#fff3bf;--bad:#ffd6d6;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--ok:#1e4d2b;--part:#5c4b0f;--bad:#5c1f1f;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid var(--line);padding:6px;vertical-align:top}}
th{{position:sticky;top:0;background:var(--bg)}}.ok{{background:var(--ok)}}.part{{background:var(--part)}}.bad{{background:var(--bad)}}
.na{{color:#888;text-align:center}}td.ok,td.part,td.bad{{text-align:center;white-space:nowrap}}code{{font-size:12px}}
</style></head><body><h1>Stage report — {html.escape(Path(eval_name).name)}</h1>
<p>Each cell = expected values still present after that stage (⚠ = a stale/wrong value is present there). Read left to right: the first non-green cell is where the answer is lost. Hover a cell for details; open <i>details</i> for the data at every stage.</p>
<ul>{funnel}</ul>
<table><tr><th>id</th><th>question</th><th>search</th><th>resolver</th><th>writer</th><th>final</th><th>score</th><th>verdict</th></tr>
{''.join(body)}</table></body></html>""")


if __name__ == "__main__":
    main()
