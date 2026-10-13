"""ANARCI/IMGT CDR annotation for the 40 VHH structures, in PDB residue numbering."""
import csv
from pathlib import Path
from anarci import run_anarci

THREE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
 "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
 "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}

def read_chain(path):
    seq, rn, ch, seen = [], [], [], set()
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        r = line[17:20].strip()
        if r not in THREE:
            continue
        k = (line[21], line[22:27])
        if k in seen:
            continue
        seen.add(k)
        seq.append(THREE[r]); rn.append(int(line[22:26])); ch.append(line[21])
    return "".join(seq), rn, ch

from anarci.germlines import all_germlines
ALPACA_V = set(all_germlines["V"]["H"].get("alpaca", {}))
HUMAN_V = set(all_germlines["V"]["H"].get("human", {}))

IMGT_CDR = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
HALLMARK = [42, 49, 50, 52]
VH_LIKE = {42: "V", 49: "G", 50: "L", 52: "W"}   # conventional VH tetrad

pdb_dir = Path("/home/yfeng17/boltzgen1/VHH/pdb")
recs = {f.stem: read_chain(f) for f in sorted(pdb_dir.glob("*.pdb"))}
seqs = [(k, v[0]) for k, v in recs.items()]

_, numbered, details, _ = run_anarci(
    seqs, scheme="imgt", ncpu=8, assign_germline=True,
    allowed_species=["alpaca", "human", "mouse", "rat", "rabbit", "rhesus", "pig"])

rows = []
for (pid, seq), num, det in zip(seqs, numbered, details):
    dom, start, end = num[0]
    d = det[0]
    rn, ch = recs[pid][1], recs[pid][2]
    # walk the numbered domain alongside the PDB residue numbers
    i = start
    imgt_to_pdb, imgt_to_aa = {}, {}
    for (idx, ins), aa in dom:
        if aa == "-":
            continue
        imgt_to_pdb.setdefault(idx, []).append(rn[i])
        imgt_to_aa.setdefault(idx, []).append(aa)
        i += 1
    row = {"pdb": pid, "chain": ch[0], "seq_len": len(seq),
           "chain_type": d["chain_type"], "species": d["species"],
           "v_gene": (d.get("germlines", {}) or {}).get("v_gene", [[None, None]])[0][1],
           "v_identity": round((d.get("germlines", {}) or {}).get("v_gene", [None, 0])[1], 3),
           "evalue": f"{d['evalue']:.1e}",
           "domain_pdb_start": rn[start], "domain_pdb_end": rn[end]}
    for name, (lo, hi) in IMGT_CDR.items():
        nums = [n for k in range(lo, hi + 1) for n in imgt_to_pdb.get(k, [])]
        aas = "".join("".join(imgt_to_aa.get(k, [])) for k in range(lo, hi + 1))
        row[f"{name}_seq"] = aas
        row[f"{name}_len"] = len(aas)
        row[f"{name}_pdb_range"] = f"{min(nums)}..{max(nums)}" if nums else ""
        row[f"{name}_contiguous"] = bool(nums) and (max(nums) - min(nums) + 1 == len(nums))
    hall = {k: "".join(imgt_to_aa.get(k, ["-"])) for k in HALLMARK}
    row["hallmark_42_49_50_52"] = "/".join(hall[k] for k in HALLMARK)
    nvh = sum(1 for k in HALLMARK if hall[k] == VH_LIKE[k])
    row["vh_like_positions"] = nvh
    # which germline DB does the assigned V gene actually come from?
    vg = row["v_gene"]
    row["germline_db"] = ("alpaca" if vg in ALPACA_V
                          else "human" if vg in HUMAN_V else "unknown")
    if row["germline_db"] == "alpaca" and nvh <= 1:
        row["classification"] = "VHH (camelid, VHH hallmark tetrad)"
    elif row["germline_db"] == "alpaca" and nvh >= 3:
        row["classification"] = "camelid single domain, VH-like tetrad (IGHV3S1 family)"
    elif row["germline_db"] == "alpaca":
        row["classification"] = "VHH (camelid, partial hallmark tetrad)"
    else:
        row["classification"] = "human/humanized single-domain VH (not camelid VHH)"
    row["sequence"] = seq
    rows.append(row)

fields = list(rows[0])
out = Path("/home/yfeng17/boltzgen1/VHH/vhh40_anarci_cdr.tsv")
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
    w.writeheader(); w.writerows(rows)

print(f"{'pdb':6} {'species':8} {'v_gene':14} {'v_id':>5} {'cdr1':>4} {'cdr2':>4} {'cdr3':>4} "
      f"{'cdr1 range':>11} {'cdr2 range':>11} {'cdr3 range':>11} {'tetrad':9} {'VHlike':>6}")
print("-"*110)
for r in rows:
    print(f"{r['pdb']:6} {r['species'][:8]:8} {str(r['v_gene'])[:14]:14} {r['v_identity']:>5} "
          f"{r['cdr1_len']:>4} {r['cdr2_len']:>4} {r['cdr3_len']:>4} "
          f"{r['cdr1_pdb_range']:>11} {r['cdr2_pdb_range']:>11} {r['cdr3_pdb_range']:>11} "
          f"{r['hallmark_42_49_50_52']:9} {r['vh_like_positions']:>6}")

noncontig = [(r['pdb'], k) for r in rows for k in ("cdr1","cdr2","cdr3") if not r[f"{k}_contiguous"]]
print(f"\nnon-contiguous CDR ranges: {noncontig or 'none'}")
print(f"chain types: {sorted({r['chain_type'] for r in rows})}")
print(f"species: { {s: sum(1 for r in rows if r['species']==s) for s in sorted({r['species'] for r in rows})} }")
print(f"VH-like tetrad (>=3 of 4 VH positions): {[r['pdb'] for r in rows if r['vh_like_positions']>=3]}")
print(f"germline DB: { {s: sum(1 for r in rows if r['germline_db']==s) for s in sorted({r['germline_db'] for r in rows})} }")
print("\nclassification:")
for c in sorted({r["classification"] for r in rows}):
    members = [r["pdb"] for r in rows if r["classification"] == c]
    print(f"  {len(members):>2}  {c}")
    if len(members) <= 5:
        print(f"      -> {', '.join(members)}")
print(f"\nwrote {out}")
