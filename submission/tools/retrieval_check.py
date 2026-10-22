#!/usr/bin/env python3
"""For each eval question, check whether a chunk with the expected values is in the top search
results. LOST means the raw page had the values but no chunk did.

    python3 submission/tools/retrieval_check.py [--eval eval/public_eval.jsonl] [-v]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402
import solution  # noqa: E402
from kb.resolver import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "eval/public_eval.jsonl"))
    ap.add_argument("--snapshots", default=str(ROOT / "snapshots"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("-v", action="store_true", help="show top results for misses")
    args = ap.parse_args()

    ks = solution.build()
    t = time.time()
    for s in load_snapshots(Path(args.snapshots)):
        ks.ingest(s)
    print(f"ingest {time.time() - t:.1f}s")
    for row in ks.ingest_log:
        print("  ", row)

    items = [json.loads(l) for l in open(args.eval) if l.strip()]
    hits = total = top1 = in5 = 0
    for it in items:
        must, must_not = it["must_include"], it["must_not_include"]
        if not must:
            continue
        sid = ks.select_snapshot(it.get("as_of"))
        ok = lambda text: all(contains(text, m) for m in must) and not any(contains(text, m) for m in must_not)
        gold = {r.content_id for r in ks.store.records if sid in r.present_in and ok(r.text)}
        all_hits = ks.search(it["question"], sid)
        results = all_hits[: args.k]
        rank = next((i + 1 for i, (r, *_) in enumerate(results) if r.content_id in gold), None)
        ev = resolve(it["question"], [(r, rel) for r, rel, _, _ in all_hits], ks.store, sid)
        top_ok = bool(ev) and ev[0].rec.content_id in gold
        in5 += any(e.rec.content_id in gold for e in ev[:5])
        total += 1
        hits += rank is not None
        top1 += top_ok
        if gold:
            status = f"hit@{rank}" if rank else "MISS"
        else:
            raw = any(ok(p.raw) for p in ks.pages[sid].values())
            # Also reached when the values are spread over several chunks.
            status = "LOST" if raw else "NOGOLD"
        print(f"{it['id']:8} {it['category']:20} {status:7} {'TOP' if top_ok else '   '} gold={len(gold):<3} {it['question'][:62]}")
        if args.v and not top_ok:
            for r, rel, cos, b in results[:3]:
                o = ks.store.occurrences_in(r, sid)[0]
                print(f"         search  cos={cos:.2f} bm25={b:5.1f} {r.chunk_type:7} {o.url_norm[23:75]}  {r.text[:70]!r}")
            for e in ev[:2]:
                print(f"         picked  tier={e.tier:+.1f} trust={e.trust:.2f} {e.date or '-':10} {e.occ.url_norm[23:75]}  {e.rec.text[:70]!r}")
    print(f"\nrecall@{args.k}: {hits}/{total} = {hits / max(1, total):.2f}   resolver top-1: {top1}/{total}   in writer's 5: {in5}/{total}")

    print("\nshould_refuse top scores (cosine, bm25):")
    for it in items:
        if it["category"] == "should_refuse":
            h = ks.search(it["question"], ks.select_snapshot(it.get("as_of")), filtered=False)[:1]
            print(f"  {it['id']} {h[0][2]:.2f} {h[0][3]:5.1f}  {it['question'][:60]}" if h else f"  {it['id']} none")


if __name__ == "__main__":
    main()
