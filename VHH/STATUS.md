# JN.1 RBD × VHH 300k run — status

As of 2026-10-18. Host `gpu21-sublab`, user `frank`, run dir `~/boltzgen1/VHH`,
output `~/boltzgen1/VHH/output`. 8× H200, shared with another user (`lee`).

## Where the run is

| Step | State |
|---|---|
| 1 design | **done** — 300,000 designs, reused in ~9 s on every launch |
| 2 inverse_folding | **done** — 300,000, reused in ~13 s |
| 3 folding | **running**, restarted 2026-10-18 at `FOLD_BATCH=15`, ~26 h, 2500 iterations/rank |
| 4 esmfold2_scoring | blocked on step 3; its blocker is already fixed (see below) |
| 5 analysis, 6 filtering | not started |

Launched with:

```bash
cd ~/boltzgen1/VHH
PYTHONWARNINGS=ignore::ResourceWarning STEPS="folding" FOLD_BATCH=15 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash sh/run_jn1_vhh40.sh output 2>&1 | tee -a output/run.log
```

## Verify before trusting this run

Both folding bugs below were caught by checks, not by reading code. Run both.

```bash
cd ~/boltzgen1/VHH
# 1. identity: does each file belong to the design it is named after?
python sh/check_folding_outputs.py \
  --design-dir output/intermediate_designs_inverse_folded --sample 60
# want: "OK: every sampled file matches the design it is named after."

# 2. count: one file per design, not one per batch
ls -fU output/intermediate_designs_inverse_folded/fold_out_npz | grep -c '\.npz$'
# want: files / (iterations x 8) == 15
```

`ls | wc -l` on these directories hangs for ~20 min — it sorts 300k entries.
Use `ls -fU` (no sort) and let `head`/`grep -c` consume it.

## Three bugs found and fixed

**`c1bc1ca` — inference coordinate augmentation.** `AtomDiffusion.sample()`
expands the batch by `multiplicity`, but asked `compute_random_augmentation`
for only `multiplicity` rotations, so einsum failed for any batch > 1.

**`ced7c06` — confidence module could not batch diffusion samples.**
`predict_step` hardcodes `run_confidence_sequentially=True`, whose path
asserted a batch of 1. Ported `confidence_chunk_size` from
`1326093445a/boltzgen`; `tests/test_confidence_multiplicity.py` shows all 15
`compute_ptms` outputs bit-identical batched vs sequential. Requires
`confidence_chunk_size == diffusion_samples`; the run script keeps them equal.

**`3728de9` + `e8227c4` — the folding writer, which cost 24.5 h.**
`FoldingWriter.write_on_batch_end` assumed a data batch of 1. It
(a) named output `batch["id"][0]` and wrote **one file per batch**, silently
discarding 14 of every 15 designs; (b) took coords at `argmax(confidence)`
over the whole batch while taking other features at index 0, pairing one
design's coordinates with another's identity; and (c) left batch padding in
the saved arrays, which `analyze_utils.get_fold_metrics` aligns element-wise
against the design's own atoms. Now splits per design via
`boltzgen.task.predict.batch_slicing` and trims padding from the saved copy
only — `Structure.from_feat` still needs padded shapes.

The first run reported `2500/2500` and `Number of failed structure
predictions: 0` while producing 20,000 wrong files out of 300,000. Iteration
counts and failure counts cannot detect this; **only the file count and the
identity check can.**

## Step 4 is already unblocked

`esmfold2_scoring` refuses a chain whose full sequence it cannot prove it has.
Both inputs (`JN1_RBD_clean.pdb`, `4QGY.pdb`) are ATOM-only with no SEQRES, so
every design's stored context says `complete=false`, and that is baked in at
design time — editing the YAML cannot reach the existing 300k.

Sidecars were generated for all 300,000 into `output/source_context`
(`sh/make_source_context_sidecars.py`). `SRC_CTX` in the run script picks that
directory up automatically. After folding:

```bash
PYTHONWARNINGS=ignore::ResourceWarning \
STEPS="esmfold2_scoring analysis filtering" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash sh/run_jn1_vhh40.sh output 2>&1 | tee -a output/run.log
```

## Settings and why

| Variable | Value | Reason |
|---|---|---|
| `DIFF_BATCH` | 5 | designs per trunk run at the design step; already spent |
| `FOLD_BATCH` | 15 | needs the fixed writer; default in-script is 1 |
| `FOLD_SAMPLES` | 1 | one structure per design; 5 would be ~77 h instead of ~26 h |
| `FOLD_RECYCLING` | 3 | matches the 200k production run in the other repo |
| `ESM_SAMPLES` | 1 | matches FOLD_SAMPLES |
| `ESM_LOOPS` | 20 | upstream default; the dominant cost in step 4 |
| `ESM_ACCEL` | auto | see open questions |

## Open questions

**Chain T is 184 residues.** An RBD construct is usually ~195–220 (spike
319–541). If `JN1_RBD_clean.pdb` is a trimmed crop rather than the expressed
construct, the sidecars now tell ESMC the protein is exactly those 184
residues, and step 4's scores shift accordingly. Consistent across all 300k so
ranking stays internally valid, but worth confirming.

**Root cause of the ATOM-only PDBs is unfixed.** Declaring `full_sequences`
for chains T and A in `yaml/vhh40_combined.yaml`, or restoring SEQRES to the
two PDBs, would stop newly generated designs needing sidecars. Needs the real
sequences, not ones inferred from ATOM records.

**`ESM_ACCEL=auto` may be doing nothing.** `auto` only applies CUDA-graph
replay up to 256 crop tokens; `fused` allows 512. A VHH plus this RBD is ~310.
Check `input_audit.crop_tokens` in any esmfold2 result JSON once step 4 runs —
if it is 257–512, `ESM_ACCEL=fused` is free throughput. It changes kernels, so
scores shift slightly; do not mix modes within one comparison.

## Quarantined, safe to delete once the new run verifies

- `~/folded_batch15_corrupt/` — 20,064 files from the dropped/mislabelled run
- `~/folded_padded_bad/` — ~2,400 files with batch padding left in

## Gotchas worth keeping

- `STEPS` is an environment variable: it goes **before** `bash`, and the
  output directory is a positional argument.
- Folding appearing to hang with idle GPUs usually means a rank crashed; grep
  the log for a traceback before assuming a deadlock. A crashed rank leaves
  the others blocked in a collective, which looks identical to a hang.
- Step 3's reuse check runs at the **start** of a step, so a writer problem
  only surfaces on the next launch — up to a day later.
