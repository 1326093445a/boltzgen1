#!/usr/bin/env python
"""Write source_context sidecars so esmfold2_scoring can run on existing designs.

esmfold2_scoring refuses a chain whose full sequence it cannot prove it has.
An ATOM-only PDB carries no SEQRES, so every chain sourced from one is stored
with complete=False, and that flag is baked into each design's metadata at
design time - changing the YAML now cannot reach designs already generated.

This rewrites each design's own stored context with complete=True, into a
directory passed as source_context_dir. It refuses any chain whose stored
indices are not a contiguous 0..n-1 run covering every parsed residue, because
only then is "the residues in the file" the same claim as "the full sequence".

READ THIS BEFORE USING IT. Declaring a chain complete tells ESMC the molecule
is exactly the residues in the PDB. That is correct for a standalone construct.
If the PDB is a CROP of a larger protein, it is wrong: ESMC will embed a
truncated sequence and the scores change. In that case supply the real sequence
via full_sequences/source_res_indices instead and regenerate.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def patch(args):
    npz_path, out_dir = args
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            if "source_context" not in d:
                return ("no_context", npz_path.name, None)
            ctx = json.loads(str(d["source_context"].item()))
    except Exception as exc:                       # noqa: BLE001
        return ("unreadable", npz_path.name, str(exc))

    if ctx.get("version") != 1:
        return ("bad_version", npz_path.name, str(ctx.get("version")))

    flipped = []
    for chain in ctx["chains"]:
        if chain["mol_type"] == 3 or chain.get("complete"):
            continue
        idx, names = chain["indices"], chain["residue_names"]
        if idx != list(range(len(names))):
            return ("not_contiguous", npz_path.name,
                    f"chain {chain['source_chain']}: "
                    f"{len(idx)} indices, {len(names)} residues, starts {idx[:3]}")
        chain["complete"] = True
        flipped.append(chain["source_chain"])

    (out_dir / f"{npz_path.stem}.json").write_text(json.dumps(ctx))
    return ("ok", npz_path.name, ",".join(flipped))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design-dir", required=True,
                   help="the dir esmfold2_scoring reads, i.e. its design_dir")
    p.add_argument("--out-dir", required=True, help="written; pass as source_context_dir")
    p.add_argument("--apply", action="store_true", help="without this, inspects 20 and stops")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    design_dir, out_dir = Path(a.design_dir), Path(a.out_dir)
    files = sorted(design_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {design_dir}")
    print(f"{len(files):,} design metadata files in {design_dir}")

    if not a.apply:
        out_dir = Path("/tmp") / f".sidecar-probe-{out_dir.name}"
        files = files[:20]
        print(f"DRY RUN: inspecting {len(files)} of them, writing nothing durable\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    tally, examples = {}, {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (status, name, detail) in enumerate(
                ex.map(patch, ((f, out_dir) for f in files), chunksize=256), 1):
            tally[status] = tally.get(status, 0) + 1
            examples.setdefault(status, (name, detail))
            if a.apply and i % 25000 == 0:
                print(f"  {i:,} / {len(files):,}", flush=True)

    print()
    for status, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        name, detail = examples[status]
        print(f"{status:15s} {n:7,}   e.g. {name}" + (f"  [{detail}]" if detail else ""))

    if set(tally) == {"ok"}:
        print(f"\nAll {tally['ok']:,} contexts patched." if a.apply else
              f"\nDry run clean. Re-run with --apply to write all {len(sorted(design_dir.glob('*.npz'))):,}.")
    else:
        raise SystemExit("\nSome designs were not patched; nothing should be trusted until that is understood.")
