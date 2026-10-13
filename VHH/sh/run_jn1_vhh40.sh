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
#   FOLD_BATCH=10        refolding batch size (structure prediction); see note
#   SPEC=                override the design spec path
#   ALPHA=               leave empty for the BoltzGen1 default
#   REUSE=true           resume completed work
#   EXTRA_ARGS=""        passed straight through to `boltzgen run`
#
# NOTE on DIFF_BATCH: BoltzGen defaults to 10 for runs >=100 designs, but all
# designs in one diffusion batch share a single YAML parse - so they share one
# framework AND one CDR-length combo AND one trunk run. At the default, 300k
# designs would be only 30k independent draws. DIFF_BATCH=1 is the deliberate
# deviation; raise it only to trade independence for throughput.
#
# NOTE on FOLD_BATCH: the folding (structure prediction) step batches designs with
# no statistical cost - by then each design is already fixed, so this is pure
# throughput. Measured peak memory on this codebase is linear in the PRODUCT of
# batch size and diffusion_samples:
#     peak_MiB ~= 4973 + 2062 * (batch * diffusion_samples)
# That model reproduced every ok/OOM outcome measured on a 24 GB RTX 4090,
# including the boundary between batch 1 and 2 at the default 5 samples.
# With the BoltzGen1 default diffusion_samples=5 on a 141 GB H200:
#     batch 13 theoretical, 12 at 10% headroom, 11 at 15%  -> 10 chosen as safe.
# Raise only after confirming on the target GPU; lower it if you hit OOM.
# Note this is an extrapolation from a 24 GB card and is conservative, since the
# fit is on nvidia-smi reserved memory rather than true allocation.
#
# NOTE on BUDGET: this produces ONE final_ranked_designs set across all 40
# frameworks. Per-framework analysis comes from all_designs_metrics.csv via
#   python scripts/assign_framework.py <out>/final_ranked_designs/all_designs_metrics.csv
# which recovers the framework and the three CDR lengths for every design.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${PKG_DIR}/output}"

SPEC="${SPEC:-${PKG_DIR}/yaml/vhh40_combined.yaml}"
NUM_DESIGNS="${NUM_DESIGNS:-300000}"
DIFF_BATCH="${DIFF_BATCH:-1}"
BUDGET="${BUDGET:-1000}"
FOLD_BATCH="${FOLD_BATCH:-10}"
REUSE="${REUSE:-true}"
ALPHA="${ALPHA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[[ -f "${SPEC}" ]] || { echo "Design spec not found: ${SPEC}" >&2; exit 1; }
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
echo "Diffusion batch:  ${DIFF_BATCH}  (independent framework+length draw per design)"
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
