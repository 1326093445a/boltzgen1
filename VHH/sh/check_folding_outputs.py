#!/usr/bin/env python
"""Check folded outputs against the designs they claim to be.

Counting files catches designs being dropped. It does not catch a file that
holds the wrong design - which is what a batch-axis mistake produces, and what
cost a 24.5 h run. So this compares each folded file with its own source
design, using the fact that designs differ in length: a VHH framework plus a
CDR-length combination gives each design a token and atom count that a
mismatched file will not reproduce.

Checks, per sampled design:
  - the folded file exists
  - its per-sample arrays have one entry per diffusion sample, not one per
    design in the batch (the failure mode: n_designs x n_samples entries)
  - its token and atom counts equal the source design's
  - any array shared with the source design is identical

Usage:
  python sh/check_folding_outputs.py --design-dir <out>/intermediate_designs_inverse_folded
"""
import argparse
import random
from pathlib import Path

import numpy as np


def _counts(arrays):
    """Token and atom counts, however this file happens to spell them."""
    out = {}
    for key in ("res_type", "mol_type", "token_index"):
        if key in arrays:
            out["tokens"] = arrays[key].shape[-1] if arrays[key].ndim == 1 \
                else arrays[key].shape[1] if arrays[key].ndim >= 2 else None
            break
    for key in ("atom_to_token", "backbone_mask", "atom_resolved_mask", "coords"):
        if key in arrays:
            a = arrays[key]
            out["atoms"] = a.shape[1] if a.ndim >= 2 else a.shape[0]
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design-dir", required=True)
    p.add_argument("--sample", type=int, default=40, help="designs to inspect")
    p.add_argument("--expect-samples", type=int, default=None,
                   help="FOLD_SAMPLES; inferred from the first file when omitted")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    design_dir = Path(a.design_dir)
    folded_dir = design_dir / "fold_out_npz"
    cif_dir = design_dir / "refold_cif"
    if not folded_dir.is_dir():
        print(f"no {folded_dir}")
        return 1

    # sample from what has been folded so far, so this works mid-run
    folded = []
    for entry in folded_dir.iterdir():
        if entry.suffix == ".npz":
            folded.append(entry.stem)
            if len(folded) >= 20000:
                break
    if not folded:
        print("no folded outputs yet")
        return 1
    print(f"{len(folded):,} folded outputs seen (capped at 20k for sampling)")

    random.seed(a.seed)
    picked = random.sample(folded, min(a.sample, len(folded)))

    n_samples = a.expect_samples
    problems, checked = [], 0
    widths = set()

    for stem in picked:
        try:
            with np.load(folded_dir / f"{stem}.npz", allow_pickle=True) as d:
                fold = {k: d[k] for k in d.files}
            with np.load(design_dir / f"{stem}.npz", allow_pickle=True) as d:
                src = {k: d[k] for k in d.files}
        except Exception as exc:                        # noqa: BLE001
            problems.append(f"{stem}: unreadable ({exc})")
            continue

        if n_samples is None:
            n_samples = len(fold["coords"])
            print(f"inferred diffusion samples per design: {n_samples}")

        # per-sample arrays must not carry other designs' entries
        for key in ("coords", "design_to_target_iptm", "design_ptm", "plddt"):
            if key in fold and len(fold[key]) != n_samples:
                problems.append(
                    f"{stem}: {key} has {len(fold[key])} entries, expected {n_samples} "
                    f"- looks like a whole batch, not one design")

        # length is a per-design fingerprint
        fc, sc = _counts(fold), _counts(src)

        # coords is the field a batch-axis mistake actually corrupts, so check
        # its own atom count rather than trusting a sibling array's
        if "coords" in fold and fold["coords"].ndim >= 2 and "atoms" in sc:
            n_atom = fold["coords"].shape[1]
            if n_atom != sc["atoms"]:
                problems.append(
                    f"{stem}: coords has {n_atom} atoms but the design has "
                    f"{sc['atoms']} - these coordinates are another design's")
        for what in ("tokens", "atoms"):
            if what in fc and what in sc and fc[what] != sc[what]:
                problems.append(
                    f"{stem}: {what} {fc[what]} in folded vs {sc[what]} in design "
                    f"- this file belongs to a different design")
        widths.add(fc.get("tokens"))

        # anything both files carry must agree
        for key in set(fold) & set(src):
            f, s = np.asarray(fold[key]), np.asarray(src[key])
            if f.dtype.kind in "OSU" or s.dtype.kind in "OSU":
                continue
            if f.shape == s.shape and not np.array_equal(f, s):
                problems.append(f"{stem}: {key} differs from the design's own value")

        if not (cif_dir / f"{stem}.cif").is_file():
            problems.append(f"{stem}: no CIF beside the NPZ")
        checked += 1

    print(f"checked {checked} designs, {len(widths)} distinct token counts among them")
    if len(widths) == 1 and checked > 5:
        print("  note: all sampled designs are the same length, so the length "
              "fingerprint proves less here than it would on a mixed sample")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems[:20]:
            print(f"  {line}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1

    print("\nOK: every sampled file matches the design it is named after.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
