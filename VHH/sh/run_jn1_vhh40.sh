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
#   FOLD_BATCH=15        refolding batch size (independent designs per forward)
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
# NOTE on FOLD_BATCH: this is data.cfg.batch_size for the folding step - the
# number of INDEPENDENT designs per forward pass. It is a different axis from
# --diffusion_batch_size, which the CLI passes through as diffusion_samples
# (several structures from one trunk run). Both multiply the work in flight:
#     structures in flight = FOLD_BATCH * diffusion_samples
# At FOLD_BATCH=15 with fold.yaml's diffusion_samples=5 that is 75.
#
# Measured on 8x H200 (143771 MiB each), FOLD_BATCH=10 x 5 samples:
#     51.5 GiB per card, 99-100% util, 624-678W of a 700W cap.
# Memory scales sub-linearly, so 15 should land near 70-75 GiB. Note the cards
# were already compute-saturated at 10, so a larger batch may not raise
# throughput - judge it on designs/second at steady state, not on the ETA shown
# during the first iterations, which includes JIT and CUDA-graph warmup.
#
# REQUIREMENT: confidence_chunk_size must equal diffusion_samples. fold.yaml
# sets both to 5, so the default works. If you override one, override both:
#     EXTRA_ARGS="--config folding diffusion_samples=N confidence_chunk_size=N"
# When confidence_chunk_size is smaller than diffusion_samples the confidence
# module takes its chunked path, which asserts a batch of 1 and makes
# FOLD_BATCH > 1 fail. That assert is also why FOLD_BATCH=10 crashed before
# the confidence_chunk_size support landed: one rank died on its first
# prediction and the rest blocked in a collective, so it looked like a hang
# rather than an error. If folding ever appears to stall with idle GPUs, grep
# the log for a traceback before assuming a deadlock.
#
# Memory is the only remaining limit and it has not been measured on an H200.
# FOLD_BATCH=1 with 5 samples uses roughly 15 GiB. Watch nvidia-smi for the
# first minute after raising it; an OOM here is immediate and loud.
#
# Parallelism also comes from the ranks: CUDA_VISIBLE_DEVICES sets the world
# size and each rank takes NUM_DESIGNS/world_size designs.
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
FOLD_BATCH="${FOLD_BATCH:-15}"
REUSE="${REUSE:-true}"
ALPHA="${ALPHA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "${SPEC}" ]] || { echo "Design spec not found: ${SPEC}" >&2; exit 1; }

if [[ "${FOLD_BATCH}" != "1" && "${EXTRA_ARGS}" == *diffusion_samples* \
      && "${EXTRA_ARGS}" != *confidence_chunk_size* ]]; then
  echo "ERROR: EXTRA_ARGS overrides diffusion_samples without setting" >&2
  echo "       confidence_chunk_size to match. With FOLD_BATCH > 1 the folding" >&2
  echo "       step will fail on its first prediction. See NOTE on FOLD_BATCH." >&2
  exit 1
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
echo "Refold batch:     ${FOLD_BATCH}  (x5 diffusion samples = $(( FOLD_BATCH * 5 )) structures in flight)"
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
