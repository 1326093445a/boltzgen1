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
#   FOLD_BATCH=1         refolding batch size; >1 CORRUPTS OUTPUT, see note
#   FOLD_SAMPLES=1       structures refolded per design; see note
#   FOLD_RECYCLING=3     recycling steps for refolding
#   ESM_SAMPLES=1        structures per design in esmfold2_scoring
#   ESM_LOOPS=20         esmfold2 trunk loops; the dominant cost, see note
#   ESM_ACCEL=auto       auto | fused | off, see note
#   STEPS=               e.g. "esmfold2_scoring analysis filtering" to resume
#   SRC_CTX=             source_context sidecars; auto-used at <out>/source_context
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
# NOTE on FOLD_BATCH: LEAVE THIS AT 1. It is data.cfg.batch_size for folding,
# and the model accepts it, but FoldingWriter cannot write the result.
#
# write_on_batch_end names its output batch["id"][0] and writes ONE file per
# batch, so (FOLD_BATCH-1)/FOLD_BATCH of every batch is discarded with no
# error and no entry in "failed structure predictions". Worse, it takes
# coords and plddt at argmax over `confidence` while taking every other
# feature at index 0; that argmax is meant to pick among diffusion samples,
# but with FOLD_BATCH>1 it ranges over designs, so the file pairs one
# design's coordinates with another's identity.
#
# Measured: FOLD_BATCH=15 for 24.5 h on 8xH200 folded all 300,000 designs and
# left 20,000 files (2500 batches x 8 ranks), each of them wrong. The step
# reported 2500/2500 and "Number of failed structure predictions: 0".
#
# Raising this is only safe once the writer handles a batch axis, and only
# with a design-by-design diff against a FOLD_BATCH=1 run to prove it.
#
# NOTE on FOLD_SAMPLES: this is diffusion_samples for the folding step - how
# many structures each design is refolded into. It multiplies the whole step.
# Measured on 8x H200 at FOLD_BATCH=10, 73.5 s per iteration with 5 samples:
#     5 samples -> 1.5M structures -> ~77 h
#     1 sample  -> 300k structures -> ~16 h
# 1 is the default, matching the 200k production run in the boltzgen repo. Use
# more only if you want a per-design spread to rank on rather than a point
# estimate. confidence_chunk_size is always set equal to it, which is the
# condition that lets FOLD_BATCH exceed 1.
#
# NOTE on ESM_SAMPLES / ESM_LOOPS: esmfold2_scoring runs on every design, with
# no pre-filter, so its cost is set the same way. ESM_SAMPLES is its
# diffusion_samples; 1 matches FOLD_SAMPLES. ESM_LOOPS is the trunk recycling
# count and is the dominant term - cost is roughly linear in it - so it is the
# first thing to lower if this step is too slow. 20 is the upstream default and
# is kept here; on a separate antibody benchmark, 3 loops versus 10 changed
# binder/non-binder AUROC by under 0.01, so the default may be far more than
# needed.
#
# NOTE on ESM_ACCEL: 'auto' only applies CUDA-graph replay to crops of at most
# 256 tokens, 'fused' raises that to 512 and enables BF16 kernels. A VHH plus
# the JN.1 RBD is around 350 tokens, so 'auto' may be silently skipping
# acceleration entirely. Check crop_tokens in any esmfold2 result JSON
# (input_audit.crop_tokens); if it is between 257 and 512, 'fused' is worth
# using. It changes kernels, so scores shift slightly - do not mix modes within
# one comparison.
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
# NOTE on SRC_CTX and STEPS: esmfold2_scoring runs ESMC over each chain's
# COMPLETE source sequence, and refuses any chain it cannot prove it has one
# for. An ATOM-only PDB carries no SEQRES, so both inputs here are stored with
# complete=false, and that flag is baked into every design's metadata at design
# time - editing the YAML cannot reach designs already generated. Symptom:
#     ValueError: Full sequence is unavailable for T (...JN1_RBD_clean.pdb)
# after folding has already finished. Fix without refolding:
#     python sh/make_source_context_sidecars.py \
#       --design-dir <out>/intermediate_designs_inverse_folded \
#       --out-dir <out>/source_context --apply --workers 32
#     STEPS="esmfold2_scoring analysis filtering" bash sh/run_jn1_vhh40.sh <out>
# <out>/source_context is picked up automatically once it exists. The generator
# refuses any chain whose residues are not a contiguous run from position 1, so
# a genuinely truncated chain is never silently declared complete; read its
# docstring before relying on it. The lasting fix is to declare full_sequences
# for both chains in the YAML, or restore SEQRES to the two PDBs, so newly
# generated designs carry a complete context.
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
FOLD_SAMPLES="${FOLD_SAMPLES:-1}"
FOLD_RECYCLING="${FOLD_RECYCLING:-3}"
ESM_SAMPLES="${ESM_SAMPLES:-1}"
ESM_LOOPS="${ESM_LOOPS:-20}"
ESM_ACCEL="${ESM_ACCEL:-auto}"
STEPS="${STEPS:-}"
# Auto-used when present; sh/make_source_context_sidecars.py writes it.
SRC_CTX="${SRC_CTX:-${OUT}/source_context}"
[[ -d "${SRC_CTX}" ]] || SRC_CTX=""
REUSE="${REUSE:-true}"
ALPHA="${ALPHA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "${SPEC}" ]] || { echo "Design spec not found: ${SPEC}" >&2; exit 1; }

if [[ "${FOLD_BATCH}" != "1" ]]; then
  echo "ERROR: FOLD_BATCH=${FOLD_BATCH}. FoldingWriter.write_on_batch_end writes ONE" >&2
  echo "       file per batch, named batch['id'][0], and mixes coordinates chosen by" >&2
  echo "       argmax over the batch with features taken from element 0. Any value" >&2
  echo "       above 1 silently drops (FOLD_BATCH-1)/FOLD_BATCH of the designs AND" >&2
  echo "       corrupts the ones it keeps. Set FOLD_BATCH=1. See NOTE on FOLD_BATCH." >&2
  exit 1
fi

if [[ "${EXTRA_ARGS}" == *diffusion_samples* || "${EXTRA_ARGS}" == *confidence_chunk_size* ]]; then
  echo "ERROR: set FOLD_SAMPLES (or ESM_SAMPLES) instead of overriding" >&2
  echo "       diffusion_samples or confidence_chunk_size through EXTRA_ARGS." >&2
  echo "       For folding the two must stay equal or the confidence module" >&2
  echo "       takes its chunked path and asserts a batch of 1." >&2
  exit 1
fi
N_FW="$(grep -c '_scaffold\.yaml' "${SPEC}" || true)"

ESM_CFG=(--config esmfold2_scoring "diffusion_samples=${ESM_SAMPLES}" "num_loops=${ESM_LOOPS}")
[[ -n "${SRC_CTX}" ]] && ESM_CFG+=("source_context_dir=${SRC_CTX}")

OPT=()
[[ "${REUSE}" == "true" ]] && OPT+=(--reuse)
[[ -n "${ALPHA}" ]] && OPT+=(--alpha "${ALPHA}")
# shellcheck disable=SC2206
# shellcheck disable=SC2206
[[ -n "${STEPS}" ]] && OPT+=(--steps ${STEPS})
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
echo "Refold batch:     ${FOLD_BATCH}  (x${FOLD_SAMPLES} diffusion samples = $(( FOLD_BATCH * FOLD_SAMPLES )) structures in flight)"
echo "Refold samples:   ${FOLD_SAMPLES} structure(s) per design, ${FOLD_RECYCLING} recycling steps"
echo "ESMFold2:         ${ESM_SAMPLES} sample(s), ${ESM_LOOPS} loops, acceleration=${ESM_ACCEL}"
echo "Source context:   ${SRC_CTX:-<none; esmfold2_scoring needs one for ATOM-only PDBs>}"
echo "Steps:            ${STEPS:-all 6}"
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
  --config folding data.cfg.batch_size="${FOLD_BATCH}" recycling_steps="${FOLD_RECYCLING}" diffusion_samples="${FOLD_SAMPLES}" confidence_chunk_size="${FOLD_SAMPLES}" \
  "${ESM_CFG[@]}" \
  --esmfold2_acceleration "${ESM_ACCEL}" \
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
