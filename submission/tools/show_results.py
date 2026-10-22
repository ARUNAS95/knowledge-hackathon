#!/usr/bin/env python3
"""Show each question of an eval run with its answer, cited pages, quotes and score.

    .venv/bin/python submission/tools/show_results.py --eval eval/my_rewordings_eval.jsonl --results results/rewordings
Writes <results>/answers.md and prints a short summary.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("eval/public_eval.jsonl"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--misses", action="store_true", help="only questions that lost points")
    args = ap.parse_args()

    questions = {q["id"]: q for q in map(json.loads, filter(str.strip, args.eval.read_text().splitlines()))}
    rows = [json.loads(l) for l in (args.results / "per_question.jsonl").read_text().splitlines() if l.strip()]

    out = [f"# Answers: {args.eval.name}\n"]
    for r in rows:
        if args.misses and r["score"] >= 1:
            continue
        q = questions.get(r["id"], {})
        a = r.get("answer") or {}
        src = {True: "right page cited", False: "wrong page cited", None: "no expected page"}[r.get("source_hit")]
        out.append(f"## {r['id']} ({r['category']}): score {r['score']}, {src}, {r['latency_s']} s")
        out.append(f"**Q:** {q.get('question', '?')}" + (f"  (as of {q['as_of']})" if q.get("as_of") else ""))
        out.append(f"**A:** {a.get('text', '')}")
        out.append(f"Expected: {q.get('must_include', [])}; must not say: {q.get('must_not_include', [])}")
        if r.get("missed") or r.get("trap_hits"):
            out.append(f"Missed: {r.get('missed')}; stale values said: {r.get('trap_hits')}")
        if q.get("expected_sources"):
            out.append(f"Expected source: {', '.join(q['expected_sources'])}")
        for s in a.get("sources", []):
            out.append(f"- {s['url']} ({s.get('snapshot_id', '')})  \n  > {s.get('quote', '')}")
        out.append("")

    path = args.results / "answers.md"
    path.write_text("\n".join(out))
    n = len(rows)
    full = sum(1 for r in rows if r["score"] >= 1)
    cited = [r for r in rows if r.get("source_hit") is not None]
    print(f"{n} questions, {full} full marks; citations: {sum(1 for r in cited if r['source_hit'])}/{len(cited)} right")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
