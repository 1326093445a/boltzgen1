#!/usr/bin/env python3
"""Build exact framework fingerprints for the combined-spec run.

Parses each per-framework wrapper YAML once and reads the retained (non-designed)
segments straight off BoltzGen's own design mask. That avoids any assumption about
residue numbering or missing backbone atoms, so the fingerprints are exact even
for files with incomplete residues (e.g. 5IVN).

Writes framework_templates.json for scripts/assign_framework.py.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import gemmi
import pandas as pd
from boltzgen.data.parse.schema import YamlDesignParser


def one(name: str) -> str:
    info = gemmi.find_tabulated_residue(name)
    return info.one_letter_code.upper() if info else "X"


def chain_and_mask(parsed):
    st, dm = parsed.structure, parsed.design_info.res_design_mask.astype(bool)
    for nm, s0, n0 in zip(st.chains["name"], st.chains["res_idx"], st.chains["res_num"]):
        if dm[s0 : s0 + n0].any():
            seq = "".join(one(r["name"]) for r in st.residues[s0 : s0 + n0])
            return seq, dm[s0 : s0 + n0]
    raise RuntimeError("no designed chain found")


def segments(seq: str, mask) -> tuple[list[str], list[int]]:
    """Split into fixed (non-designed) segments and designed run lengths."""
    fixed, runs, cur, run = [], [], [], 0
    for c, d in zip(seq, mask):
        if d:
            if cur:
                fixed.append("".join(cur)); cur = []
            run += 1
        else:
            if run:
                runs.append(run); run = 0
            cur.append(c)
    if cur: fixed.append("".join(cur))
    if run: runs.append(run)
    return fixed, runs


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-dir", type=Path, default=here.parent / "yaml")
    ap.add_argument("--out", type=Path, default=here.parent / "framework_templates.json")
    a = ap.parse_args()

    moldir = next((Path.home() / ".cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots").glob("*/mols.zip"))
    parser = YamlDesignParser(mol_dir=moldir)
    mols: dict = {}
    man = pd.read_csv(a.yaml_dir / "manifest.csv").set_index("pdb")

    templates = []
    for pdb, row in man.iterrows():
        wrapper = a.yaml_dir / f"job_{pdb}_wrapper.yaml"
        fixed = None
        # a couple of parses: confirm the fixed segments are invariant to the random lengths
        for _ in range(3):
            seq, mask = chain_and_mask(parser.parse_yaml(wrapper, mol_dir=moldir, mols=mols))
            f, _runs = segments(seq, mask)
            if fixed is None:
                fixed = f
            elif fixed != f:
                raise RuntimeError(f"{pdb}: fixed segments not invariant across parses")
        ranges = [tuple(int(x) for x in rg.split("..")) for rg in str(row.insert_range).split()]
        templates.append({"pdb": pdb, "fixed": fixed, "ranges": [list(r) for r in ranges]})
        print(f"  {pdb}: {len(fixed)} fixed segments, lens {[len(x) for x in fixed]}, ranges {ranges}")

    # fingerprint uniqueness
    fp = {"|".join(t["fixed"]): t["pdb"] for t in templates}
    print(f"\nunique fingerprints: {len(fp)} / {len(templates)}")
    if len(fp) != len(templates):
        raise SystemExit("fingerprints are not unique")
    a.out.write_text(json.dumps(templates, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
