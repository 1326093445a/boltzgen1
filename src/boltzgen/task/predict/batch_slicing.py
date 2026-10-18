"""Splitting a folding batch back into per-design pieces.

Two leading-axis conventions share a folding prediction. Per-design features
(residue types, atom maps, masks) carry an axis of ``n_designs``; coordinates,
pLDDT and the confidence metrics carry one entry per (design, diffusion
sample), laid out design-major. Confusing the two does not raise - it mislabels
output - so the arithmetic lives here on its own and is tested directly.

Batching also pads: every design in a batch is extended to the longest one in
it. A batch of one has nothing to pad to, so files written that way hold exact
lengths, and downstream analysis aligns folded coordinates against the design's
own atoms element-wise. Padding therefore has to come back off before writing,
or the extra rows silently shift every RMSD and lDDT.
"""

import numpy as np


def design_spans(n_designs: int, n_samples: int, confidence) -> list[tuple[int, int, int]]:
    """Per design, its ``(lo, hi, best)`` indices into the (design, sample) axis.

    ``lo:hi`` is that design's block of samples; ``best`` is the single index of
    its highest-confidence sample, chosen *within* the design so one design's
    coordinates can never be attributed to another.
    """
    scores = np.asarray(confidence).reshape(n_designs, n_samples)
    return [
        (d * n_samples, (d + 1) * n_samples, d * n_samples + int(np.argmax(scores[d])))
        for d in range(n_designs)
    ]


def take(value, design: int, lo: int, hi: int, n_designs: int, n_samples: int):
    """One design's slice of ``value``, keeping the shape a batch of one gives.

    Arrays on the (design, sample) axis keep all of that design's samples;
    arrays on the design axis keep a leading dimension of one; anything whose
    length matches neither is shared by the whole batch and passes through.
    """
    count = len(value)
    if count == n_designs * n_samples:
        return value[lo:hi]
    if count == n_designs:
        return value[design : design + 1]
    return value


def pick(value, design: int, lo: int, best: int, n_designs: int, n_samples: int):
    """One design's single entry of ``value``, with the leading axis dropped.

    Used for the structure written to CIF: coordinates and pLDDT come from the
    chosen sample, per-design features from this design, and anything shared
    from the front.
    """
    count = len(value)
    if count == n_designs * n_samples:
        return value[best]
    if count == n_designs:
        return value[design]
    return value[0]


def real_extent(pad_mask, design: int, n_designs: int):
    """How many atoms or tokens this design really has, behind the pad mask."""
    if pad_mask is None:
        return None
    mask = np.asarray(pad_mask)
    if mask.ndim >= 1 and mask.shape[0] == n_designs:
        return int(mask[design].sum())
    if mask.ndim == 1:
        return int(mask.sum())
    return None


def trim(value, real_atoms, real_tokens, ref_atom_len, ref_tokens_len):
    """Drop batch padding from one design's array, for the copy written to disk.

    Any axis as long as the padded atom extent is cut to this design's real
    atom count, and likewise for tokens. The reference lengths come from the
    data itself (coordinates for atoms, residue types for tokens) rather than
    from the mask, so an axis that merely happens to match a mask length is
    left alone.

    Only the saved copy is trimmed. The structure handed to Structure.from_feat
    keeps its padded shapes, because it is read alongside batch-derived masks
    such as atom_pad_mask that are still padded.
    """
    if not isinstance(value, np.ndarray) or value.ndim == 0:
        return value
    out = value
    for axis in range(out.ndim):
        length = out.shape[axis]
        if real_atoms is not None and ref_atom_len is not None \
                and length == ref_atom_len and real_atoms < length:
            index = [slice(None)] * out.ndim
            index[axis] = slice(0, real_atoms)
            out = out[tuple(index)]
        elif real_tokens is not None and ref_tokens_len is not None \
                and length == ref_tokens_len and real_tokens < length:
            index = [slice(None)] * out.ndim
            index[axis] = slice(0, real_tokens)
            out = out[tuple(index)]
    return out
