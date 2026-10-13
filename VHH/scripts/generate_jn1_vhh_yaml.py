"""Generate JN1 VHH design YAMLs: interface-proximal CDR design mask, fixed anchors,
per-CDR length range anchored to each scaffold's native ANARCI length."""
import argparse
from pathlib import Path
import pandas as pd

# Anchors held native (identity + structure), expressed as positions within each CDR.
# Derived from antigen-contact frequency over 19 VHH-antigen complexes:
#   CDR1 pos1 = 11% contact | CDR2 pos1 = 5% | CDR3 pos1/2 = 5%/0%, last = 26%
ANCHOR_N = {1: 1, 2: 1, 3: 2}   # residues fixed at the CDR N-terminal stem
ANCHOR_C = {1: 0, 2: 0, 3: 1}   # residues fixed at the CDR C-terminal stem
WIDTH    = {1: 1, 2: 1, 3: 2}   # +- length range on the designed window

def parse_rng(txt):
    lo, hi = str(txt).split("..")
    return int(lo), int(hi)

def windows(row):
    """-> {cdr: (cdr_lo, cdr_hi, win_lo, win_hi, win_len, lo_len, hi_len)}"""
    out = {}
    for c in (1, 2, 3):
        lo, hi = parse_rng(row[f"cdr{c}_pdb_range"])
        wlo, whi = lo + ANCHOR_N[c], hi - ANCHOR_C[c]
        wlen = whi - wlo + 1
        w = WIDTH[c]
        out[c] = (lo, hi, wlo, whi, wlen, max(1, wlen - w), wlen + w)
    return out

def fmt_len(lo, hi):
    return str(lo) if lo == hi else f"{lo}..{hi}"

def scaffold_yaml(row, fixed=None):
    """fixed: dict {cdr: absolute designed-window length} for a single-combo job."""
    w = windows(row)
    pid = row["pdb"]
    design = ",".join(f"{w[c][2]}..{w[c][3]}" for c in (1, 2, 3))
    L = ["# Auto-generated JN.1 RBD VHH job for %s." % pid,
         "# Interface-proximal CDR design mask; CDR stems held native as anchors.",
         "# Native ANARCI/IMGT CDR lengths: %d/%d/%d  (full CDRs: %s, %s, %s)"
         % (row.cdr1_len, row.cdr2_len, row.cdr3_len,
            row.cdr1_pdb_range, row.cdr2_pdb_range, row.cdr3_pdb_range),
         "# Designed windows (anchors excluded): %s" % design,
         "path: ../pdb/VHH/%s.pdb" % pid,
         "include:", "  - chain:", "      id: %s" % row["chain"], "",
         "design:", "  - chain:", "      id: %s" % row["chain"],
         "      res_index: %s" % design, "",
         "structure_groups:", "  - group:", "      id: %s" % row["chain"],
         "      visibility: 2",
         "  - group:", "      id: %s" % row["chain"], "      visibility: 0",
         "      res_index: %s" % design, "",
         "exclude:"]
    for c in (1, 2, 3):
        L += ["  - chain:", "      id: %s" % row["chain"],
              "      res_index: %d..%d" % (w[c][2], w[c][3])]
    L += ["", "design_insertions:"]
    for c in (1, 2, 3):
        n = fmt_len(*(([fixed[c]] * 2) if fixed else (w[c][5], w[c][6])))
        L += ["  - insertion:", "      id: %s" % row["chain"],
              "      res_index: %d" % w[c][2], "      num_residues: %s" % n]
    L += ["", "reset_res_index:", "  - chain:", "      id: %s" % row["chain"], ""]
    return "\n".join(L)

WRAPPER = """# Wrapper: JN.1 RBD target (hotspot S158,Y159,F161,T164,G166,V167) + {pid} VHH scaffold.
entities:
  - file:
      path: ../pdb/Target/JN1_RBD_clean.pdb
      include:
        - chain:
            id: T
      binding_types:
        - chain:
            id: T
            binding: 158,159,161,164,166,167

  - file:
      path: {scaffold}
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="/home/yfeng17/boltzgen1/VHH/vhh40_anarci_cdr.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    df = pd.read_csv(a.tsv, sep="\t")
    if a.only:
        df = df[df.pdb.isin(a.only)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        w = windows(r)
        s = out / f"{r.pdb}_scaffold.yaml"
        s.write_text(scaffold_yaml(r))
        (out / f"job_{r.pdb}_wrapper.yaml").write_text(
            WRAPPER.format(pid=r.pdb, scaffold=s.name))
        rows.append({
            "pdb": r.pdb,
            "native": f"{r.cdr1_len}/{r.cdr2_len}/{r.cdr3_len}",
            "design_windows": ",".join(f"{w[c][2]}..{w[c][3]}" for c in (1,2,3)),
            "window_len": "/".join(str(w[c][4]) for c in (1,2,3)),
            "insert_range": " ".join(f"{w[c][5]}..{w[c][6]}" for c in (1,2,3)),
            "final_cdr_len_range": " ".join(
                f"{w[c][5]+ANCHOR_N[c]+ANCHOR_C[c]}-{w[c][6]+ANCHOR_N[c]+ANCHOR_C[c]}"
                for c in (1,2,3)),
            "n_combos": (w[1][6]-w[1][5]+1)*(w[2][6]-w[2][5]+1)*(w[3][6]-w[3][5]+1),
        })
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    print(man.to_string(index=False))
    print(f"\n{len(man)} scaffolds -> {out}   total combos: {man.n_combos.sum()}")
