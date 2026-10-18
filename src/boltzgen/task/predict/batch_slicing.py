"""Splitting a folding batch back into per-design pieces.

Two leading-axis conventions share a folding prediction. Per-design features
(residue types, atom maps, masks) carry an axis of ``n_designs``; coordinates,
pLDDT and the confidence metrics carry one entry per (design, diffusion
sample), laid out design-major. Confusing the two does not raise - it mislabels
output - so the arithmetic lives here on its own and is tested directly.
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
