"""Which IMGT CDR positions contact antigen, measured on the original depositions."""
import json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, gemmi
from anarci import run_anarci
warnings.filterwarnings("ignore")

PDB_DIR = Path("/home/yfeng17/boltzgen1/VHH/pdb/VHH")
ROOT = Path("/home/yfeng17/boltzgen")
CUT = 4.5
IMGT_CDR = {1: (27, 38), 2: (56, 65), 3: (105, 117)}

ids = sorted(p.stem for p in PDB_DIR.glob("*.pdb"))
raw = {i: next((ROOT / "Round2").rglob(f"{i}_raw.cif")) for i in ids}

def chain_seq(ch):
    rs = [r for r in ch if r.find_atom("CA", "*")]
    return "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                   if gemmi.find_tabulated_residue(r.name) else "X" for r in rs), rs

results, no_antigen = {}, []
for pid in ids:
    st = gemmi.read_structure(str(raw[pid])); st.setup_entities()
    st.remove_alternative_conformations(); st.remove_hydrogens()
    st.remove_ligands_and_waters()
    model = st[0]
    chains = {}
    for ch in model:
        s, rs = chain_seq(ch)
        if len(s) >= 20:
            chains[ch.name] = (s, rs)
    if not chains:
        continue
    # run ANARCI on every chain; VHH = chains ANARCI numbers as H
    vhh, other = {}, []
    for n, (s, rs) in chains.items():
        try:
            out = run_anarci([(n, s)], scheme="imgt", ncpu=1)
            num = out[1][0]
        except Exception:
            num = None
        if num is not None:
            vhh[n] = num[0]
        else:
            other.append(n)
    if not vhh:
        continue
    # pick the VHH copy with the most antigen neighbours later; start with first
    ag_atoms = []
    for n in other:
        for r in chains[n][1]:
            for a in r:
                ag_atoms.append(a.pos.tolist())
    if not ag_atoms:
        no_antigen.append(pid)
        continue
    AG = np.array(ag_atoms)

    best = None
    for vn, (dom, start, end) in vhh.items():
        rs = chains[vn][1]
        i = start
        pos_contacts, pos_present = {}, set()
        ncontact = 0
        for (idx, ins), aa in dom:
            if aa == "-":
                continue
            res = rs[i]; i += 1
            cdr = next((k for k, (lo, hi) in IMGT_CDR.items() if lo <= idx <= hi), None)
            if cdr is None:
                continue
            P = np.array([a.pos.tolist() for a in res])
            d = np.sqrt(((P[:, None, :] - AG[None, :, :]) ** 2).sum(-1)).min()
            pos_present.add((cdr, idx))
            if d <= CUT:
                pos_contacts[(cdr, idx)] = float(d)
                ncontact += 1
        if best is None or ncontact > best[0]:
            best = (ncontact, vn, pos_contacts, pos_present)
    results[pid] = {"chain": best[1], "n_contact_cdr_pos": best[0],
                    "contacts": {f"{c}:{i}": round(d, 2) for (c, i), d in best[2].items()},
                    "present": sorted(f"{c}:{i}" for c, i in best[3]),
                    "antigen_chains": other}

print(f"complexes with antigen: {len(results)} | no antigen in deposition: {len(no_antigen)}")
print(f"  no antigen: {', '.join(no_antigen)}\n")
print(f"{'pdb':6} {'vhh':4} {'antigen chains':16} {'CDR pos in contact':>18}  per-CDR contacts")
print("-" * 86)
for pid, r in sorted(results.items()):
    per = defaultdict(int)
    for k in r["contacts"]:
        per[int(k.split(":")[0])] += 1
    print(f"{pid:6} {r['chain']:4} {','.join(r['antigen_chains'])[:16]:16} {r['n_contact_cdr_pos']:>18}  "
          f"CDR1 {per[1]:>2}  CDR2 {per[2]:>2}  CDR3 {per[3]:>2}")
Path("paratope.json").write_text(json.dumps(results, indent=2))
