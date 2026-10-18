"""A batch of N designs must be split exactly as N batches of one would be.

The folding writer mixes a per-design axis with a per-(design, sample) axis.
Getting that wrong does not raise: with FOLD_BATCH=15 it silently wrote one
file per batch instead of fifteen, and paired one design's coordinates with
another's identity. This pins the equivalence.

Run: python tests/test_folding_writer_batch.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boltzgen.task.predict.batch_slicing import design_spans, pick, take

N_DESIGN, N_SAMPLE, N_TOKEN = 4, 3, 5


def _arrays(n_design, n_sample, rng):
    """One array on each axis, plus one shared by the whole batch."""
    n_bm = n_design * n_sample
    return {
        "coords": rng.normal(size=(n_bm, N_TOKEN, 3)),          # design x sample
        "plddt": rng.uniform(size=(n_bm, N_TOKEN)),             # design x sample
        "design_to_target_iptm": rng.uniform(size=n_bm),        # design x sample
        "design_ptm": rng.uniform(size=n_bm),                   # design x sample
        "res_type": rng.integers(0, 20, size=(n_design, N_TOKEN)),   # design
        "atom_to_token": rng.uniform(size=(n_design, N_TOKEN, N_TOKEN)),  # design
        "shared_lut": rng.uniform(size=(7, 2)),                  # neither
    }


def _split(arrays, n_design, n_sample):
    """What the writer stores and draws, per design."""
    conf = 0.8 * arrays["design_to_target_iptm"] + 0.2 * arrays["design_ptm"]
    out = {}
    for d, (lo, hi, best) in enumerate(design_spans(n_design, n_sample, conf)):
        stored = {k: take(v, d, lo, hi, n_design, n_sample) for k, v in arrays.items()}
        drawn = {
            k: (v[best] if k in ("coords", "plddt")
                else pick(v, d, lo, best, n_design, n_sample))
            for k, v in arrays.items()
        }
        out[d] = (stored, drawn, best - lo)
    return out

def _test_padding():
    """A padded batch of different-length designs must save unpadded arrays.

    Batching pads every design to the longest in the batch. A batch of one
    never pads, and analysis re-featurises each design alone and compares
    shapes, so the saved copy has to come back to the design's real extent.
    """
    from boltzgen.task.predict.batch_slicing import real_extent, trim

    lengths = [(297, 2376), (316, 2528), (304, 2432)]     # (tokens, atoms)
    pad_tok = max(t for t, _ in lengths)
    pad_atom = max(a for _, a in lengths)
    rng = np.random.default_rng(7)

    atom_pad_mask = np.zeros((len(lengths), pad_atom), dtype=bool)
    token_pad_mask = np.zeros((len(lengths), pad_tok), dtype=bool)
    for d, (n_tok, n_atom) in enumerate(lengths):
        atom_pad_mask[d, :n_atom] = True
        token_pad_mask[d, :n_tok] = True

    ok = True
    print(f"\n{'design':8s} {'tokens':>14s} {'atoms':>14s} {'padding gone':>13s}")
    print("-" * 54)
    for d, (n_tok, n_atom) in enumerate(lengths):
        saved = {
            "coords": rng.normal(size=(1, pad_atom, 3)),
            "res_type": rng.integers(0, 20, size=(1, pad_tok)),
            "atom_to_token": rng.uniform(size=(1, pad_atom, pad_tok)),
            "design_ptm": rng.uniform(size=1),            # no sequence axis
        }
        real_atoms = real_extent(atom_pad_mask, d, len(lengths))
        real_tokens = real_extent(token_pad_mask, d, len(lengths))
        res_type = saved["res_type"]
        while res_type.ndim > 1 and res_type.shape[0] == 1:
            res_type = res_type[0]
        out = {
            k: trim(v, real_atoms, real_tokens, saved["coords"].shape[-2],
                    res_type.shape[0])
            for k, v in saved.items()
        }
        got_tok = out["res_type"].shape[-1]
        got_atom = out["coords"].shape[-2]
        good = (got_tok == n_tok and got_atom == n_atom
                and out["atom_to_token"].shape == (1, n_atom, n_tok)
                and out["design_ptm"].shape == (1,))
        ok &= good
        print(f"{d:<8d} {f'{got_tok} (want {n_tok})':>14s} "
              f"{f'{got_atom} (want {n_atom})':>14s} {str(good):>13s}")
    print("-" * 54)
    print(f"batch padded to {pad_tok} tokens / {pad_atom} atoms; "
          f"untrimmed output would mis-align every RMSD")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arrays = _arrays(N_DESIGN, N_SAMPLE, rng)

    batched = _split(arrays, N_DESIGN, N_SAMPLE)

    # the same designs, each submitted alone
    singles = {}
    for d in range(N_DESIGN):
        lo, hi = d * N_SAMPLE, (d + 1) * N_SAMPLE
        alone = {
            k: (v[lo:hi] if len(v) == N_DESIGN * N_SAMPLE
                else v[d : d + 1] if len(v) == N_DESIGN else v)
            for k, v in arrays.items()
        }
        singles[d] = _split(alone, 1, N_SAMPLE)[0]

    print(f"{'design':8s} {'stored equal':>13s} {'drawn equal':>12s} {'best sample':>12s}")
    print("-" * 50)
    ok = True
    for d in range(N_DESIGN):
        b_stored, b_drawn, b_best = batched[d]
        s_stored, s_drawn, s_best = singles[d]
        same_stored = all(np.array_equal(b_stored[k], s_stored[k]) for k in arrays)
        same_drawn = all(np.array_equal(b_drawn[k], s_drawn[k]) for k in arrays)
        ok &= same_stored and same_drawn and b_best == s_best
        print(f"{d:<8d} {str(same_stored):>13s} {str(same_drawn):>12s} "
              f"{b_best:>7d} vs {s_best}")
        for k in arrays:
            if not np.array_equal(b_stored[k], s_stored[k]):
                print(f"    stored differs: {k}")
            if not np.array_equal(b_drawn[k], s_drawn[k]):
                print(f"    drawn  differs: {k}")

    # the chosen sample must come from this design's own block, never a neighbour's
    conf = 0.8 * arrays["design_to_target_iptm"] + 0.2 * arrays["design_ptm"]
    for d, (lo, hi, best) in enumerate(design_spans(N_DESIGN, N_SAMPLE, conf)):
        if not lo <= best < hi:
            print(f"    design {d}: best index {best} outside its block [{lo},{hi})")
            ok = False

    # the pre-fix behaviour, to show the test discriminates
    old_best = int(np.argmax(conf))
    print("-" * 50)
    print(f"single argmax over the whole batch would give index {old_best} "
          f"(design {old_best // N_SAMPLE}) for every design")
    ok &= _test_padding()

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
