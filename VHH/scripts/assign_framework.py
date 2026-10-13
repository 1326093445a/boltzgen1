#!/usr/bin/env python3
"""Recover the source VHH framework (and per-CDR designed lengths) for designs
produced from the combined spec yaml/vhh40_combined.yaml.

The combined spec picks a scaffold at random per design, so every design shares
one output id stem. The non-designed residues stay native, which makes
"native minus the three design windows" a unique fingerprint per framework.

A design chain must decompose exactly as
    A + X1 + B + X2 + C + X3 + D
where A/B/C/D are that framework's fixed native segments around the three design
windows and X1/X2/X3 are the designed CDR segments, each with a length inside
that framework's allowed insertion range. The decomposition is solved exactly,
so this also recovers the three designed CDR window lengths.

Usage:
  python assign_framework.py all_designs_metrics.csv -o labelled.csv
  python assign_framework.py all_designs_metrics.csv --manifest ../yaml/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json

import pandas as pd


def load_templates(path: Path) -> list[dict]:
    """Load fingerprints written by build_framework_templates.py."""
    raw = json.loads(Path(path).read_text())
    out = []
    for t in raw:
        A, B, C, D = t["fixed"]
        out.append({"pdb": t["pdb"], "A": A, "B": B, "C": C, "D": D,
                    "ranges": [tuple(r) for r in t["ranges"]]})
    return out


def decompose(seq: str, t: dict):
    """Return (l1, l2, l3) designed window lengths, or None if seq is not from t."""
    A, B, C, D = t["A"], t["B"], t["C"], t["D"]
    if not seq.startswith(A) or not seq.endswith(D):
        return None
    mid = seq[len(A) : len(seq) - len(D)] if D else seq[len(A) :]
    (r1a, r1b), (r2a, r2b), (r3a, r3b) = t["ranges"]
    for l1 in range(r1a, r1b + 1):
        if mid[l1 : l1 + len(B)] != B:
            continue
        p = l1 + len(B)
        for l2 in range(r2a, r2b + 1):
            if mid[p + l2 : p + l2 + len(C)] != C:
                continue
            q = p + l2 + len(C)
            l3 = len(mid) - q
            if r3a <= l3 <= r3b:
                return (l1, l2, l3)
    return None


def assign(seq: str, templates: list[dict]):
    hits = []
    for t in templates:
        d = decompose(seq, t)
        if d is not None:
            hits.append((t["pdb"], d))
    if len(hits) == 1:
        return hits[0][0], hits[0][1], "unique"
    if not hits:
        return None, None, "no_match"
    return hits[0][0], hits[0][1], f"ambiguous:{len(hits)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="all_designs_metrics.csv from the combined run")
    ap.add_argument("-o", "--out", type=Path, default=None)
    here = Path(__file__).resolve().parent
    ap.add_argument("--templates", type=Path,
                    default=here.parent / "framework_templates.json")
    ap.add_argument("--seq-col", default="designed_chain_sequence")
    a = ap.parse_args()

    templates = load_templates(a.templates)
    df = pd.read_csv(a.csv)
    if a.seq_col not in df.columns:
        raise SystemExit(f"column {a.seq_col!r} not in {a.csv} (have: {list(df.columns)[:12]}...)")

    # cache: identical sequences resolve once
    cache: dict[str, tuple] = {}
    recs = []
    for s in df[a.seq_col].astype(str):
        if s not in cache:
            cache[s] = assign(s, templates)
        recs.append(cache[s])
    df["framework"] = [r[0] for r in recs]
    df["assigned_cdr_lengths"] = [str(list(r[1])) if r[1] else "" for r in recs]
    df["assignment"] = [r[2] for r in recs]

    out = a.out or a.csv.with_name(a.csv.stem + "_with_framework.csv")
    df.to_csv(out, index=False)

    n = len(df)
    vc = df.assignment.value_counts()
    print(f"{n} designs")
    for k, v in vc.items():
        print(f"  {k:14} {v:>8}  ({v/n:.2%})")
    print(f"\nframeworks seen: {df.framework.nunique()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
