"""compute_ptms must give the same answer batched as one sample at a time.

Run: python tests/test_confidence_multiplicity.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import torch
from boltzgen.model.layers.confidence_utils import compute_ptms
from boltzgen.data import const

torch.manual_seed(0)
B, N, M, BINS = 1, 24, 5, 64
APT = 4                      # atoms per token
A = N * APT

PROT = const.chain_type_ids["PROTEIN"]

atom_to_token = torch.zeros(B, A, N)
for t in range(N):
    atom_to_token[0, t * APT:(t + 1) * APT, t] = 1.0

feats = {
    "token_pad_mask": torch.ones(B, N),
    "atom_pad_mask": torch.ones(B, A),
    "asym_id": torch.cat([torch.zeros(B, N // 2), torch.ones(B, N - N // 2)], 1).long(),
    "mol_type": torch.full((B, N), PROT).long(),
    "atom_to_token": atom_to_token,
    # first chain is the design, second the target
    "design_mask": torch.cat([torch.ones(B, N // 2), torch.zeros(B, N - N // 2)], 1),
    "chain_design_mask": torch.cat([torch.ones(B, N // 2), torch.zeros(B, N - N // 2)], 1),
    "frames_idx": torch.stack([
        torch.stack([torch.arange(N), torch.arange(N), torch.arange(N)], -1)
    ]).long(),
}

# distinct coordinates and logits per diffusion sample
x_preds = torch.randn(B * M, A, 3) * 10.0
logits = torch.randn(B * M, N, N, BINS)

batched = compute_ptms(logits, x_preds, feats, M)

names = ["ptm", "iptm", "ligand_iptm", "protein_iptm", "chain_pair_iptm",
         "design_to_target_iptm", "design_residue_iptm", "design_iptm",
         "design_iiptm", "target_ptm", "design_ptm", "design_ipsae_min",
         "design_to_target_ipsae", "target_to_design_ipsae", "chain_pair_ipsae"]
assert len(names) == len(batched), (len(names), len(batched))

seq = [compute_ptms(logits[i:i+1], x_preds[i:i+1], feats, 1) for i in range(M)]

print(f"{'metric':26s} {'max|batched - sequential|':>26s}   verdict")
print("-" * 70)
worst = 0.0
for k, name in enumerate(names):
    bt = batched[k]
    if isinstance(bt, dict):
        d = 0.0
        n = 0
        for c1, inner in bt.items():
            items = inner.items() if isinstance(inner, dict) else [(None, inner)]
            for c2, v in items:
                ref = torch.stack([
                    (s_[k][c1][c2] if c2 is not None else s_[k][c1]).reshape(-1)
                    for s_ in seq
                ])
                d = max(d, (v.reshape(M, -1) - ref).abs().max().item())
                n += 1
        worst = max(worst, d)
        print(f"{name:26s} {d:26.3e}   {'OK' if d < 1e-4 else '*** MISMATCH ***'}  ({n} chain pairs)")
        continue
    bt = bt.reshape(M, -1)
    sq = torch.stack([s[k].reshape(-1) for s in seq])
    d = (bt - sq).abs().max().item()
    worst = max(worst, d)
    print(f"{name:26s} {d:26.3e}   {'OK' if d < 1e-4 else '*** MISMATCH ***'}")

print("-" * 70)
print(f"worst: {worst:.3e}  ->  {'PASS' if worst < 1e-4 else 'FAIL'}")
