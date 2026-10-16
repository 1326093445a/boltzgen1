#!/bin/bash
# JN.1 RBD x 40 VHH frameworks - ONE combined design spec, BoltzGen1 defaults.
#
# yaml/vhh40_combined.yaml lists all 40 scaffold YAMLs in a single `path:` list.
# BoltzGen re-parses the spec per design and picks one scaffold at random
# (schema.py:1946-1949), so every design independently draws:
#   * a framework   - uniform over the 40 (verified: chi-sq 41.5 on 39 df)
#   * CDR1/CDR2/CDR3 window lengths - uniform over native +-1/+-1/+-2 (45 combos)
#
# NUM_DESIGNS is the TOTAL for this form (one input spec), not per framework.
# At 300000 each framework gets ~7500 designs (+-86), ~167 per length combo.
#
# Pipeline is the BoltzGen1 default for nanobody-anything:
#   design -> inverse_folding -> folding -> esmfold2_scoring -> analysis -> filtering
#
# Usage:
#   bash sh/run_jn1_vhh40.sh output
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh/run_jn1_vhh40.sh output
#
# Env vars:
#   NUM_DESIGNS=300000   total designs across all frameworks
#   DIFF_BATCH=1         see note; 1 keeps every design an independent draw
#   BUDGET=1000          size of the final diversity-optimised set (global)
#   FOLD_BATCH=1         refolding batch size; >1 is unvalidated, see note
#   SPEC=                override the design spec path
#
# Output: pass it as the first argument, absolute path recommended:
#   bash sh/run_jn1_vhh40.sh /data/jn1_vhh40_300k
# Default is <VHH>/output. Budget ~150+ GiB of disk for 300k designs
# (~0.5 MiB/design for design + inverse-folded + refold artifacts, plus
# esmfold2 and analysis outputs on top).
#   ALPHA=               leave empty for the BoltzGen1 default
#   REUSE=true           resume completed work
#   EXTRA_ARGS=""        passed straight through to `boltzgen run`
#
# NOTE on DIFF_BATCH: all designs in one diffusion batch share a single YAML
# parse - so they share one framework, one CDR-length combo AND one trunk run,
# differing only by diffusion noise. multiplicity = num_designs/DIFF_BATCH is
# therefore the number of INDEPENDENT (framework, length) draws:
#     DIFF_BATCH   parses    draws/framework   draws/combo   designs/combo
#          1       300,000        7,500             167           167
#          5        60,000        1,500              33           167
#         10        30,000          750              17           167
# Designs per combo is unchanged; what shrinks is independent draws, and with
# correlated batch-mates the effective n per combo shrinks with it.
# 5 is chosen as the speed/resolution compromise: framework rankings stay strong
# (1,500 draws each) while the trunk cost is amortised 5x. Use 1 if combo-level
# rankings matter most, 10 if throughput dominates.
#
# NOTE on FOLD_BATCH: 1 is the default because it is the only value this
# pipeline has actually been run at. Larger values are worth testing but are
# not yet validated.
#
# Until the confidence_chunk_size work landed, >1 could not work at all:
# predict_step calls the model with run_confidence_sequentially=True, and that
# path asserted a data batch of 1. FOLD_BATCH=10 crashed the folding step on its
# first prediction, and because one rank dying leaves the others blocked in a
# collective it reported as a silent hang rather than an error.
#
# With confidence_chunk_size == diffusion_samples that assertion is no longer
# reached (multiplicity > chunk_size is false, so the chunked branch is skipped)
# and the batched path expands feats to batch * multiplicity correctly. So >1 is
# now plausible. It is still untested, and batch-1 assumptions remain elsewhere
# in the codebase - rmsd_computation.py and several places in
# model/models/boltz.py - so raise it stepwise and check the metrics:
#     FOLD_BATCH=2, confirm all_designs_metrics.csv matches a batch-1 run,
#     then 4, then higher. Folding is the long pole, so this is worth doing.
#
# An earlier version of this note gave peak_MiB ~= 4973 + 2062 * (batch *
# diffusion_samples) and recommended batch 10. That extrapolation was from a
# 24 GB card and predated the assertion being discovered; treat it as untrusted
# until re-measured on the target GPU.
#
# Parallelism also comes from the ranks: CUDA_VISIBLE_DEVICES sets the world
# size and each rank takes NUM_DESIGNS/world_size designs. At batch 1 with
# diffusion_samples=5 folding uses roughly 15 GiB per card.
#
# NOTE on confidence_chunk_size: fold.yaml sets it to 5, matching
# diffusion_samples, so the confidence module runs once per design instead of
# once per diffusion sample. Pass
#     EXTRA_ARGS="--config folding confidence_chunk_size=0"
# to restore the old per-sample behaviour, which is useful for checking that the
# two agree on a handful of designs. Note that 0 also re-enables the batch-1
# assertion, so it cannot be combined with FOLD_BATCH > 1.
#
# NOTE on BUDGET: this produces ONE final_ranked_designs set across all 40
# frameworks. Per-framework analysis comes from all_designs_metrics.csv via
#   python scripts/assign_framework.py <out>/final_ranked_designs/all_designs_metrics.csv
# which recovers the framework and the three CDR lengths for every design.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Output dir: first positional arg, default <VHH>/output. Relative paths are
# resolved against the CURRENT directory, so they are made absolute here to
# avoid landing somewhere unexpected. Expect ~150+ GiB for a 300k run, so point
# this at a disk with room - ideally outside the git repo.
OUT="${1:-${PKG_DIR}/output}"
case "${OUT}" in
  /*) ;;
  *) OUT="$(pwd)/${OUT}" ;;
esac
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"

SPEC="${SPEC:-${PKG_DIR}/yaml/vhh40_combined.yaml}"
NUM_DESIGNS="${NUM_DESIGNS:-300000}"
DIFF_BATCH="${DIFF_BATCH:-5}"
BUDGET="${BUDGET:-1000}"
FOLD_BATCH="${FOLD_BATCH:-1}"
REUSE="${REUSE:-true}"
ALPHA="${ALPHA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "${SPEC}" ]] || { echo "Design spec not found: ${SPEC}" >&2; exit 1; }

if [[ "${FOLD_BATCH}" != "1" ]]; then
  echo "NOTE: FOLD_BATCH=${FOLD_BATCH}. Values >1 are unvalidated; they need" >&2
  echo "      confidence_chunk_size == diffusion_samples (fold.yaml sets this)." >&2
  echo "      If folding appears to hang with idle GPUs, a rank has crashed -" >&2
  echo "      check the log for a traceback before assuming a deadlock." >&2
fi
N_FW="$(grep -c '_scaffold\.yaml' "${SPEC}" || true)"

OPT=()
[[ "${REUSE}" == "true" ]] && OPT+=(--reuse)
[[ -n "${ALPHA}" ]] && OPT+=(--alpha "${ALPHA}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && OPT+=(${EXTRA_ARGS})

echo "========================================"
echo "JN.1 RBD x VHH frameworks - combined spec"
echo "========================================"
echo "Spec:             ${SPEC}"
echo "Frameworks:       ${N_FW} (picked at random per design)"
echo "Total designs:    ${NUM_DESIGNS}"
if [[ "${N_FW}" -gt 0 ]]; then
  echo "Per framework:    ~$(( NUM_DESIGNS / N_FW ))"
  echo "Per length combo: ~$(( NUM_DESIGNS / (N_FW * 45) ))  (45 combos each)"
fi
echo "Diffusion batch:  ${DIFF_BATCH}  -> $(( NUM_DESIGNS / DIFF_BATCH )) independent framework+length draws"
echo "Final set budget: ${BUDGET} (global across frameworks)"
echo "Refold batch:     ${FOLD_BATCH}  (x5 diffusion samples = $(( FOLD_BATCH * 5 )) structures in flight; must be 1)"
echo "Hotspot:          chain T 158,159,161,164,166,167"
echo "Output:           ${OUT}"
echo ""

set -x
boltzgen run "${SPEC}" \
  --output "${OUT}" \
  --protocol nanobody-anything \
  --num_designs "${NUM_DESIGNS}" \
  --diffusion_batch_size "${DIFF_BATCH}" \
  --budget "${BUDGET}" \
  --config folding data.cfg.batch_size="${FOLD_BATCH}" \
  "${OPT[@]}"
set +x

METRICS="${OUT}/final_ranked_designs/all_designs_metrics.csv"
echo ""
echo "========================================"
echo "Run complete. Results: ${OUT}/final_ranked_designs"
echo "========================================"
if [[ -f "${METRICS}" ]]; then
  echo "Assigning frameworks..."
  python "${PKG_DIR}/scripts/assign_framework.py" "${METRICS}"
else
  echo "No ${METRICS} yet; once present run:"
  echo "  python ${PKG_DIR}/scripts/assign_framework.py ${METRICS}"
fi
