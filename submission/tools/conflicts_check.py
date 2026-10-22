#!/usr/bin/env python3
"""Print every conflict conflicts() reports.

    .venv/bin/python submission/tools/conflicts_check.py [-v]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

from run_eval import load_snapshots  # noqa: E402
import solution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=str(ROOT / "snapshots"))
    ap.add_argument("-v", action="store_true", help="also show resolution and sources")
    args = ap.parse_args()

    ks = solution.build()
    for s in load_snapshots(Path(args.snapshots)):
        ks.ingest(s)
    t = time.time()
    conflicts = ks.conflicts()
    print(f"{len(conflicts)} conflicts in {time.time() - t:.2f}s\n")
    for i, c in enumerate(conflicts, 1):
        print(f"{i}. {c.topic}")
        print(f"   values: {c.values}")
        print(f"   reason: {c.reason}")
        if args.v:
            print(f"   resolution: {c.resolution}")
            for s in c.sources:
                print(f"   - {s.snapshot_id} {s.url}\n       {s.quote}")
        print()


if __name__ == "__main__":
    main()
