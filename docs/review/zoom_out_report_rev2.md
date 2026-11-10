# Goldfish Re-Audit Report — Stage 5 (Initial Zoom Out), Revision 2

**Target:** `E:\ML mini-project\docs\architecture-improvement-plan.md` (lines 1-348)

**Prior report:** `E:\ML mini-project\docs\review\zoom_out_report.md` (19 findings: B1, B2, M1-M13, N1-N4)

**Standard applied (`egm-zoom-out`):** `resources/document_standards/zoom-out-standard.md`

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-zoom-out` | completed (persona resolved from `agents/zoom_out.md`) |
| `egm-inventive-catalyst` | completed (persona resolved from `agents/inventive_catalyst.md`; the `agents/egm-inventive-catalyst.md` path does not exist in the installed plugin) |

Both reviewers explored the repository and verified document claims against code. Two
things could not be verified with read-only tools: **git history** (so item 7's
"absent from every commit in history — verified" rests on the author's own check) and
nothing else material. The synthesizer independently re-verified the two new Blockers
below that turn on code behaviour (`train.py:47-62`, the installed Keras
`RandomContrast` source, and the holdout PNG/JPEG file counts).

## Resolution Summary

| Status | Count |
| --- | --- |
| ✅ Resolved | 11 |
| ◐ Partial | 8 |
| ○ Open | 0 |

**Caveat on the count.** No prior finding was *ignored* — revision 2 engages all 19.
But B1's substance is still open in fact: `eval_holdout.py` does not exist, so every
acceptance criterion in the document remains unexecutable. "Partial" there means
"correctly planned, not yet done", and the plan around it has acquired defects of its
own (NB4).

## Severity Breakdown — NEW findings only

| Severity | Count |
| --- | --- |
| 🔴 Blocker | 5 |
| 🟡 Major | 10 |
| 🟢 Nit | 14 |

## Lens Summaries

**`egm-zoom-out`.** The renumbering into execution order is a real improvement and the
right spine (measure → free experiments → prerequisites → retrain). Scope is still
well-judged for a student mini-project, and §2 (Simplicity/YAGNI) and §4 (Leverage
Existing) now pass on substance: the bootstrap machinery became a deletion, Platt and the
FFT branch are declined with reasons, and three real repo utilities are named for reuse.
The remaining failures are §0 (Context Sufficiency) and §3 (Right Placement) — the
document's numeric spine is still unverifiable and now carries a wrong confidence-interval
claim on top of it, and item 1, declared a prerequisite for everything, is specified in a
way that cannot deliver three of the four jobs assigned to it.

**`egm-inventive-catalyst`.** Revision 2 fixed the numbers and left the seams. The unit
error became a unit table instead of a single-convention pipeline; the inconsistent
accuracies became a pinned table instead of a reproducible procedure; the threshold defect
became a bullet in one section and a deletion in another; and the one confound large
enough to explain both the grain bypass and part of the headline 88% — a third of the AI
class being PNG while none of the real class is — is unexamined in both the training
corpus and the holdout.

**Convergence.** Both lenses reached the same one-line verdict from opposite directions:
*the plan is now structurally right and still not executable.* Revision 2 corrected
almost every fact the prior review raised and, in the same pass, wrote new wrong facts
into a new table.

---

## 1. Resolution Table

| # | Prior finding (short) | Status | Evidence / what is still missing |
| --- | --- | --- | --- |
| **B1** | Measurement harness outside repo | ◐ **Partial** | Now item 1 at position 1; the misleading "`scratchpad/` under the repo" phrasing is gone (line 68). All three named reuse points are real: `calibrate_thresholds.py:30` `predict_paths`, `data.py:16` `iter_image_files`, `inference.py:32` `load_calibration`. Line 70's claim that no committed code reads `data/holdout_*` is true (grep for `holdout` across `*.py` returns nothing). **The script still does not exist**, and the item's own spec cannot deliver what it promises — see NB4. |
| **B2** | Noise σ in wrong units | ✅ **Resolved** | Conversion at lines 184-195 is correct: `preprocess_input` is `x/127.5 - 1`, applied at `data.py:26` before the augmenter at `train.py:47`. 4/127.5 = 0.0314 → 0.031 ✓, 8/127.5 = 0.0627 → 0.063 ✓, 16/127.5 = 0.1255 → 0.125 ✓. Back-conversion also right: 0.03 × 127.5 = 3.825 → "σ≈3.8 pixel units" ✓. σ=16 now explicitly tested (line 209). Successors in NB1, NB4, NM6. |
| **M1** | Invert threshold bootstrap to a deletion | ◐ **Partial** | Adopted at lines 314-321 and every supporting fact verified: `thresholds.json:10` carries the `tuning_note` stating 0.50 and 0.55 are tied; `calibrate_thresholds.py:80` clips to `[0.35, 0.65]`; `thresholds.json:5-6` show `validation_accuracy == validation_accuracy_default_0_5 == 0.893`, which only occurs on the guard branch — and that branch forces `best_threshold = 0.5`, so the file's `0.55` cannot have been written by that function. Two defects: it is filed under a heading defined at line 303 as "explicitly declined", and it contradicts item 7 bullet 1 (NM1). |
| **M2** | Threshold fitted on single-pass scores | ◐ **Partial** | Defect verified and the citation is now right: `train.py:219` fits on `predict_probabilities` (`train.py:104-111`, one pass) while serving (`inference.py:77-78`) and `calibrate_thresholds.py:46-49` flip-average two. Demoted to an item 7 bullet. Not adequate — not because it deserves a higher slot, but because the document now prescribes two incompatible fixes for the same code (NM1), and the fix is scheduled after every retrain it invalidates (NM8). |
| **M3** | Unfrozen layer count is 45, not 30 | ✅ **Resolved** | `train.py:192` is exactly `unfreeze_for_fine_tuning(model, trainable_layers=45)`; `model.py:45` is exactly `trainable_layers: int = 30`, unused. Status table line 42 and item 6a lines 244-248 both correct, including the "already run, never recorded" point. |
| **M4** | 710 phone photos already in the corpus | ◐ **Partial** | Count exact: `data/processed/real/phone_real_IMG_*` = 710; `data/processed/fake/*phone*` = 0. Burst caveat real and stated (lines 128-130); the listing also shows many non-Burst near-duplicates (`20191116_171032/171034/171035/171036`), so distinct scenes are far below 710. **Dropped without comment:** the prior report's cheaper half — hold the 710 out *during the retrain that is happening anyway*, which buys a clean real-side holdout at zero collection and zero extra compute. Line 130 defers it ("if they are ever held out"), so item 2c settles for a ceiling number when a clean one was free. |
| **M5** | Missing cell is non-square AI images | ✅ **Resolved** | Item 5 lines 227-238 adopts the confound, names the predicted asymmetric failure, and reduces the format axis to a ~20-line synthetic transform. `data/processed/fake` is 23,944 files, all numeric `NNNN.jpg`/`.png`, consistent with "contains none". A larger confound of the same family is still missed — NB2. |
| **M6** | `load_dataset` is eager and fully resident | ✅ **Resolved** | `data.py:29-56` confirmed fully resident. Arithmetic right: 48,707 × 224 × 224 × 3 × 4 = 2.933 × 10¹⁰ = 29.3 GB. Item 3c's inference also right: 1,000 test support under `train.py:32-45` (0.3 then 0.5 → 70/15/15) implies a ~6,670 corpus; 48,707/6,670 = 7.3 → "about sevenfold" ✓. The `IMAGE_SIZE` sub-point (`config.py:15`) is moot rather than unaddressed, since revision 2 drops the resize question. Remediation options are not interchangeable — NM7. |
| **M7** | Pin the headline numbers | ◐ **Partial** | Row E verifies exactly against `artifacts/metrics.json`: accuracy 0.888 (lines 2, 28), fake recall 0.8895833 → 89.0%, real recall 0.8865385 → 88.7%, support 1000 (line 33), threshold 0.5 (line 4). The table is internally arithmetic-consistent (A: 126+138 = 264/300 = 88.0%; B: 71+80 = 151/200 = 75.5%; C: 264+265 = 529/600 = 88.17 → 88.2%; D: 642+690 = 1332/1600 = 83.25 → 83.2%), and row C matches item 4's σ=0 row exactly. But it carries four new defects: a wrong confidence interval (NB5), a wrong methodology label on row E (NM3), a wrong README attribution with an incomplete reconciliation list (NM4), and no sampling rule, so it cannot be regenerated (NM2). |
| **M8** | Retrain overwrites hand-shaped artifacts | ◐ **Partial, and newly made worse** | Every fact in item 3b verified: `train.py:221` writes `THRESHOLD_PATH`; `train.py:224` → `evaluate_and_save_metrics` → `train.py:148` writes `METRICS_PATH`; `train.py:135-146` emits a nested schema while the on-disk `metrics.json:1-4` is flat; `find_best_threshold` (`calibrate_thresholds.py:81-88`) never writes `tuning_note`. But revision 2 gated all of item 3 behind line 139 — the one branch that doesn't need it — see NB3. |
| **M9** | Item 5 over-scoped; `confidence` is not a confidence | ✅ **Resolved** | Platt out of scope with a correct monotonicity argument (310-313). `index.html:369` is exactly `` return `${(value * 100).toFixed(2)}%`; `` and is the only `toFixed` in the file; `predict.py:21-22` uses `:.2%`; `inference.py:84-88` is the distance-to-threshold rescale; `inference.py:13-14` and `calibrate_thresholds.py:83-84` duplicate threshold and 0.10 band; `main.py:23` is `@app.on_event("startup")`. Residue: line 287 defers the deletion decision, and blast radius is understated (N8). |
| **M10** | Frequency branch should be an explicit won't-do | ✅ **Resolved** | Lines 305-308, with the "invalidates all four non-problems" reason, foreshadowed at lines 58-60. |
| **M11** | No-retrain transforms (grain-floor, noise TTA) | ◐ **Partial** | Both exist as 2a/2b (110-120) and are correctly motivated; the `inference.py:77` two-variant-stack claim is accurate. But: 2a does not say where in the pipeline the grain goes, whether it is per-channel, clipped, or seeded; 2b does not state that it invalidates the threshold too; the gate at lines 132-133 has no numeric bar and does not actually branch item 4; neither names the serving-nondeterminism liability. See NM9. |
| **M12** | Item 4 vs spectral branch conflict | ◐ **Partial** | Named at lines 266-269, which is progress. But the paragraph only half-prescribes ("feed any spectral path the un-augmented tensor, or keep the two separate"), declares none of the secondary liabilities the prior report asked for (two artifacts and 2× inference cost in a project that chose MobileNetV2-224 to stay cheap), and line 268 is grammatically broken (N2). Naming a contradiction is not resolving it. |
| **M13** | Cheap pre-resize spectral probe | ✅ **Resolved** | Item 6b (250-261) is a faithful statement, and the liability at 259-260 is correct: `data.py:59-64` `load_image_for_inference` decodes at full resolution (`cv2.imdecode`, line 61) then immediately calls `preprocess_image`, which resizes at `data.py:23` — the full-resolution array is discarded inside that one call and never returned. |
| **N1** | `gemini_key.txt` already answered | ✅ **Resolved** (author's pushback upheld — see §4) | `.gitignore:6` is exactly `gemini_key.txt`; the file exists on disk; no `*.py` matches `gemini\|genai\|GEMINI`. The repository-grounds claim is correct. Mildly over-corrected — N13. |
| **N2** | Altitude drop in item 1 (Keras block) | ✅ **Resolved** | The 8-line `Sequential` block is gone, replaced by prose at 197-200. The removal created NM6. |
| **N3** | Broken cross-reference, scratchpad path | ✅ **Resolved** | Line 68 says "temporary directory outside the repository"; line 52's resolution finding and line 60's reference both point correctly. |
| **N4** | Numbering vs suggested order | ✅ **Resolved** | Line 7 states execution order, there is no second list, the apologia is deleted. The conditional at line 139 is a legitimate dependency, not a competing ordering — its weakness is the undefined gate (NM9), not the structure. |

### Line-reference audit

All 17 references checked. **Exact:** `train.py:192`, `train.py:47`, `train.py:221`,
`train.py:135`, `data.py:16`, `data.py:26`, `data.py:29`,
`calibrate_thresholds.py:30`, `calibrate_thresholds.py:80`,
`calibrate_thresholds.py:83`, `inference.py:13`, `model.py:45`, `index.html:369`,
`.gitignore:6`. **Off by a line or two (pointer-to-caller, harmless):** `train.py:219`
(the fit is :220), `train.py:224` (the write is :148), `inference.py:84` (the rescale is
:85-87). **One genuine misattribution:** the `best < default + 0.02` guard is at
`calibrate_thresholds.py:75-78`, not :80 (which is the clip). See N3.

---

## 2. New Findings (Blockers → Major → Nits)

### NB1. `RandomContrast(0.1)` is clipping half the dynamic range of every augmented training batch, and lines 202-203 mis-describe it as a units question — Severity: 🔴 Blocker

* **Reviewer Lens:** `egm-inventive-catalyst`; mechanism independently re-verified by the synthesizer
* **Affected Scope:** item 4 "Change" (lines 197-203); `train.py:47-54`; `data.py:22-26`
* **Discrepancy Description:** The installed Keras `RandomContrast.__init__` defaults
  `value_range=(0, 255)` (`.venv/Lib/site-packages/keras/src/layers/preprocessing/image_preprocessing/random_contrast.py:59`)
  and `transform_images` **clips its output to that range** (same file, lines 104-106).
  `train.py:52` constructs it with no `value_range`, and `data.py:26` has already mapped
  tensors to [-1, 1]. The transform is `(x - mean) * factor + mean`; with x in [-1, 1] the
  channel mean is ≈0, so the output stays in [-1, 1] and the clip to [0, 255] clamps
  **every value below zero — everything darker than mid-grey — to 0.0**. This is not a
  "unit hazard", it is a data-destroying bug sitting in the exact block item 4 proposes to
  edit, and it contaminates the clean-accuracy baseline that item 3c establishes and item
  4's acceptance test is measured against. Worse, the two layers the document pairs it with
  are innocent: `RandomZoom` (`train.py:51`) and `RandomRotation` (`train.py:50`) default to
  `fill_mode="reflect"` and have no absolute-range dependence at all. So lines 202-203 send
  the author to check two harmless layers with a description that will not find the real
  fault in the third.
* **Rationale & Recommendation:** The document accepts that preprocessing lives in the data
  loader and that the augmentation pipeline must therefore be translated into [-1, 1] — then
  pays for that choice with a conversion table (184-195), a hazard-check to-do (202-203), a
  separate unit instruction in item 1 (91-92), and a wrong claim about which layers are
  affected. **Dissolve it instead of paying it:** take `preprocess_input` out of `data.py:26`
  and make it the model's first layer in `build_model` (`model.py:21-30`). The data path then
  carries 0-255 end to end — item 4's σ is in pixel units natively, the conversion table
  becomes unnecessary, item 1's "0-255 before preprocessing" becomes the project's only
  convention, `RandomContrast`'s default `value_range` becomes correct, and item 6c's
  "drop-in" claim becomes true (NM10). One change retires four pieces of unit bookkeeping.
  **Secondary liabilities that must be declared:** it changes the saved model's input
  contract, so it can only land *with* the retrain (items 3/4), not as a tidy-up before it;
  `artifacts/ai_image_detector.keras` becomes incompatible; and `load_image_for_inference`
  (`data.py:59-64`) and `predict_paths` (`calibrate_thresholds.py:30-52`) must change in the
  same commit or serving silently double-preprocesses. If the author prefers the minimum
  diff, the one-line version is `RandomContrast(0.1, value_range=(-1, 1))` — but the unit
  bookkeeping stays, and so do NB4, NM6, NM7 and NM10.

### NB2. A class-correlated file-format confound, larger than the one item 5 names, present in the holdout, free to both test and fix — Severity: 🔴 Blocker

* **Reviewer Lens:** `egm-inventive-catalyst`; counts independently re-verified by the synthesizer
* **Affected Scope:** item 5 (213-238); "Measured non-problems" line 50; item 4 in full; pinned rows A, C, D
* **Discrepancy Description:** Measured on disk: `data/processed/fake` holds **7,903 PNG**
  of ~23,944 (33%); `data/processed/real` holds **0 PNG** of 24,763. The same asymmetry is in
  the evaluation set: `data/holdout_newdb/fake/*.png` = **267** of 999 (27%);
  `data/holdout_newdb/real/*.png` = **0** of 1,000. A third of the AI class therefore arrives
  with no JPEG quantisation signature at all, and none of the real class does. That is the
  *same cue family* as the grain bypass item 4 spends the entire retrain on — "absence of
  acquisition noise ⇒ AI" — except it is baked into the labels by dataset construction. It is
  also in the holdout, so it inflates rows A, C and D, which are the numbers every acceptance
  test in the document is measured against. Item 5 (227-231) identifies a shape/source
  confound affecting 710 of 24,763 real files (2.9%); this one affects 33% of one class and 0%
  of the other, and the document does not mention it anywhere. The non-problems bullet at
  line 50 does not cover it: re-compressing an already-mixed *test* set measures robustness to
  an input transform, not the removal of a label-correlated cue from *training*.
* **Rationale & Recommendation:** Two steps, both cheap. (a) **Measure it free**, exactly
  like item 2c — score the fake-PNG and fake-JPEG subsets separately with item 1's script; a
  filename filter. If the PNG subset is detected far better, the cue is confirmed.
  (b) **Dissolve it** — re-encode the 7,903 PNG fakes to JPEG at a quality drawn from the real
  class's distribution before the retrain, so both classes share one save path. That removes
  the shortcut rather than training the model to ignore it, and it makes item 4's noise
  augmentation defensible instead of load-bearing. **Secondary liabilities:** re-encoding
  destroys any genuine PNG-only generator artifact, so keep the originals and treat the
  re-encoded corpus as a second dataset; and accuracy will very likely **drop** on rows A-D
  after the fix, because part of the current 88% is the shortcut. That drop is a correction,
  not a regression, and the document must say so in advance or the item will read as a failure.
  **This has to be decided before item 3c's re-baseline, not after** — it changes the meaning
  of every clean-accuracy number downstream.

### NB3. Item 3b is gated behind the one branch that does not need it, and the branch that does need it destroys the artifact — Severity: 🔴 Blocker

* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** line 139 vs lines 115-116 vs lines 149-157
* **Discrepancy Description:** Item 3 is prefixed "Only needed if item 2 does not dissolve
  item 4" (line 139). But item 2a says in its own text that it "would invalidate the 0.55
  threshold, so re-measure both" (115-116). Re-measuring means running
  `calibrate_thresholds.py`, whose `--output` defaults to `THRESHOLD_PATH`
  (`calibrate_thresholds.py:95`) — so it overwrites `artifacts/thresholds.json`, and
  `find_best_threshold` never emits `tuning_note` (`calibrate_thresholds.py:81-88`). That
  destroys exactly the value the document itself flags at 154-155 as "a value no code in the
  repo can regenerate". **So in the success branch of item 2 — the branch the plan is hoping
  for — the snapshot is skipped and the irreplaceable artifact is lost.**
* **Rationale & Recommendation:** Pull 3b out of item 3 and make it unconditional, before
  anything writes to `artifacts/`. This is pure reordering: no new content, no new code, and
  strictly smaller than any alternative. Name a snapshot destination while there —
  the existing `artifacts/metrics_before_retrain_<timestamp>.json` convention is right
  there to copy. Note also N11: `artifacts/` is gitignored, so there is no git copy to
  recover from.

### NB4. Item 1's specified interface cannot produce three of the four things item 1 claims to produce, and its acceptance criterion points at a row it cannot reproduce — Severity: 🔴 Blocker

* **Reviewer Lens:** `egm-zoom-out` and `egm-inventive-catalyst` (independently, with different sub-points; merged)
* **Affected Scope:** lines 76-99, 91-92, 94-96; `calibrate_thresholds.py:30-52`; `data.py:22-26`, `data.py:59-64`
* **Discrepancy Description:** Line 94 asserts the script is "item 4's acceptance test, item
  2's measuring instrument, the regenerator of the pinned-numbers table, and… the smoke test
  item 7 asks for". Checked one at a time:
  * **The seam does not exist.** Lines 91-92 require noise "in 0-255 pixel units before
    preprocessing". But `predict_paths` calls `load_image_for_inference(path.read_bytes())`
    (`calibrate_thresholds.py:35`), and `load_image_for_inference` (`data.py:59-64`) decodes,
    resizes *and* calls `preprocess_input` in one call with no hook. Same for `--jpeg` and
    `--scale`, which must act on bytes or on the decoded BGR array. `--noise/--jpeg/--scale`
    cannot be implemented on top of the three reuse points the document lists. The document
    *notices this exact missing seam* for item 6b (259-260) and never connects it to item 1,
    which needs the same seam first.
  * **"Before preprocessing" does not pin the point, and the difference is ~4×.**
    `preprocess_image` (`data.py:22-26`) does resize → BGR2RGB → `preprocess_input` in one
    call. Injecting σ=16 before the 1024→224 resize and after it are not the same experiment:
    area/bilinear downsampling by ~4.5× attenuates i.i.d. noise by roughly that factor, so
    pre-resize σ=16 reaches the network as single-digit σ. Item 4's training noise is
    unambiguously post-resize (`train.py:59-62` operates on already-224 tensors). So item 4's
    acceptance test ("σ=16 recall rises above 70%") may be evaluating a materially weaker
    perturbation than the one the model was trained against — **a 4× unit mismatch, the exact
    mistake revision 2 is proud of having caught.**
  * **Items 2a and 2b are not expressible with one `--noise` flag.** 2a is "attack at σ=16,
    then normalise with σ=8 before scoring" — two noise stages, one knob. 2b is "score at σ=0
    and σ=8 and average" — a change to the *scoring function*, not the input. As specified,
    item 1 cannot measure item 2 at all, and item 2 is what the plan runs second.
  * **Item 2c needs a filename filter, not a folder.** `eval_holdout.py <folder>` cannot score
    the 710 `phone_real_IMG_*` files, which sit inside `data/processed/real/` alongside 24,053
    others.
  * **The acceptance criterion points at a row the script cannot reproduce.** Lines 98-99 say
    "running it on `data/holdout_newdb` reproduces row A within its confidence interval". Row A
    is 300 images served over HTTP; `data/holdout_newdb` holds 1,000 real + 999 fake (verified).
    The offline baseline for that set is row C (600 images, 88.2%). Running on all 1,999
    produces a sixth number that is in no row. The gate is also ±CI, which per NB5 is ±3.7
    points for row A — anything in 84.3-91.7 passes, which cannot detect the preprocessing
    regressions item 7's smoke test exists to catch.
  * **Labels come from nowhere.** The reuse list (85-88) omits the one function that solves
    both labelling and determinism: `calibrate_thresholds.py:19-27` `balanced_sample` already
    draws a seeded (`SEED = 42`, `config.py:19`), balanced, *labelled* path list from two class
    folders — precisely the "N (N/2+N/2)" structure every row in the table has.
    `iter_image_files` alone returns no labels.
* **Rationale & Recommendation:** Three sentences fix the spec. (1) Say where the transform
  attaches — "`predict_paths` gains an optional transform callback applied between
  `cv2.imdecode` and `preprocess_image`", or split `preprocess_image` into
  `decode`/`resize`/`normalise`. If NB1's resolution is adopted, this question disappears.
  (2) State the injection point explicitly: "σ in 0-255 units applied to the decoded image
  **after** resize to 224, before `preprocess_input`" — the only choice that matches the
  training augmenter. (3) Add `--per-class N --seed S`, adopt `balanced_sample`, re-derive
  row A as the seeded 150+150 draw, and make the acceptance test
  **exact-match-on-rerun** (same seed ⇒ byte-identical number) plus a sanity band against
  row C. Determinism is a stronger and cheaper gate than a confidence interval. Also add
  `--filter GLOB` for item 2c, and state which of 2a/2b is a scorer change.

### NB5. The confidence-interval claim is arithmetically wrong, and item 4's acceptance tolerance is below the document's own noise floor — Severity: 🔴 Blocker

* **Reviewer Lens:** `egm-zoom-out` (Blocker), `egm-inventive-catalyst` (Major) — merged at the higher severity; arithmetic independently re-verified by the synthesizer
* **Affected Scope:** lines 25-26, 172, 206-207
* **Discrepancy Description:** Normal-approximation 95% CIs on accuracy from the stated
  counts: row A, n=300, p=0.880 → **±3.68**; row B, n=200, p=0.755 → **±5.96**; row C, n=600,
  p=0.882 → **±2.58**; row D, n=1600, p=0.832 → **±1.83**. The stated "±2 to ±3.5 points"
  (line 25) is wrong at both ends and understates row B by about 70%. The per-class columns
  are worse, because each is computed on half the images: row A's 84.0% AI-recall is ±5.9 on
  150; row B's 71.0% is ±8.9 on 100; row C's 55.7% σ=16 recall is ±5.6 on 300. The blanket
  sentence covers all of those columns and applies to none of them. This is not pedantry,
  because the document then uses that floor as a **decision rule** ("differences smaller than
  that are not differences", line 26) and immediately contradicts it at line 207: item 4
  passes only if "clean accuracy stays within 1.5 points of the item 3c re-baseline". At the
  available sample sizes a 1.5-point tolerance is unmeasurable — a true 3-point regression
  will pass, and a true no-change will fail, roughly at random. **The plan's single most
  expensive item has an undecidable pass/fail gate.** The "monotonic and significant from σ=4"
  claim at line 172 is unsupported for the same reason.
* **Rationale & Recommendation:** The resolution is free, and it is the right one anyway:
  these are **paired** comparisons on the same images, where the relevant interval is on the
  *difference* and is much tighter than either marginal CI. State that convention once at
  line 26 and both item 4's 1.5-point bound and the σ=4 significance claim become defensible;
  omit it and a careful reader rejects both. Then correct the marginal CIs per row rather
  than a blanket band, and give the per-class columns their own.

---

### NM1. Three mutually inconsistent prescriptions for the threshold subsystem — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both, independently)
* **Affected Scope:** item 7 bullet 1 (275-278) vs out-of-scope (314-321) vs 115, 149-155
* **Discrepancy Description:** In one document: item 7 bullet 1 says **fix** the threshold fit
  at `train.py:219`; out-of-scope says **delete** the fitter ("the correct move is a deletion,
  not an addition… remove the search"); item 2a says the threshold must be **re-measured**
  after grain normalisation, 2b implies the same without saying it, item 3b says the frozen
  value must survive a retrain, and item 4's retrain overwrites it via `train.py:221`. If the
  deletion happens, bullet 1 is work on deleted code and 2a has no instrument left to
  re-measure with. If bullet 1 happens, "the correct move is a deletion" is false. The author
  cannot execute both and the document does not choose. The deletion also breaks `train.py:12`
  (import), `train.py:220` (call) and `train.py:221` (write) — the frozen value then has to be
  written by *something*, and the document does not say what. **This contradiction did not
  exist in revision 1; it was created by splitting prior M1 and M2 into two different sections.**
* **Rationale & Recommendation:** One decision, stated once. The lazy and more final option:
  delete `find_best_threshold` and `train.py:219-221`, freeze 0.55 citing `thresholds.json:10`,
  and add a one-off `--fit-threshold` mode to item 1's script for the cases (2a, 2b, the
  retrain) that genuinely need a re-fit. Net deletion plus one flag; removes the search loop as
  a drift source permanently; and gives 2a the instrument it currently lacks. **Liability:** the
  frozen value then lives in a committed artifact with no generator, so item 7's smoke test
  must assert the *served* threshold, or a future retrain silently reverts it again — precisely
  the failure `artifacts/metrics_before_retrain_*.json` suggests has already happened once.

### NM2. The pinned table pins values but not procedures, so it cannot be regenerated — and the utility that would fix it is already in the repo — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** lines 13-14, 19-22, 34, 98-99
* **Discrepancy Description:** Row A uses 300 of 1,999 available `holdout_newdb` images, row B
  200 of 600 hard-mined, row C 600 of 1,999, row D 1,600 of 2,599. No seed, no selection rule,
  no class split for row D. Specifically:
  * **Row D has no mix ratio.** It merges `holdout_newdb` (easy) with `holdout_hard_mined`
    (deliberately hard) into "1600", so its 83.2% is a pure function of an unstated ratio. This
    is the row the README quotes recall and specificity from.
  * **Row B is model-selected, not held out in the usual sense.** The filenames carry the
    mining scores: `data/holdout_hard_mined/fake/hard_fake_0092_0.2793.jpg`,
    `…/real/hard_real_0067_0.7997.jpg`. The set was mined against the model being evaluated, so
    after item 4's retrain row B is **not regenerable** by item 1's script and is not comparable
    to its own former value. The table presents it as a stable benchmark.
  * **Rows A and C are unspecified subsamples of the same dataset at the same threshold and
    they disagree** by 4.0 points of AI-recall (84.0 vs 88.0) and 3.7 of specificity.
  Line 34 makes one script "the sole producer of this table"; it cannot be, because nothing
  records which images each row used. And `calibrate_thresholds.py:19-27` `balanced_sample`
  already does seeded balanced sampling — item 1 lists two of its neighbours from the same file
  for reuse and omits it.
* **Rationale & Recommendation:** Pin the *procedure* alongside the value: one footnote per row
  giving the selection rule, the class split and the scoring mode, and mark rows whose
  regeneration requires more than running item 1's script (B needs re-mining; E needs the old
  `train.py`). Simpler still, and what a regenerable table actually looks like: drop the
  subsample rows and pin the full-set numbers. Until item 1 exists, label rows A-D
  "unreproducible pending item 1" rather than bolding them as authority.

### NM3. Row E is labelled flip-averaged; it cannot be — and its support is unbalanced — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** lines 14-15, 23, 28-29
* **Discrepancy Description:** The header says "All rows use flip-averaged scoring at threshold
  0.55 unless the row says otherwise". Row E overrides the threshold (0.50) but not the scoring
  method, so it inherits "flip-averaged". `metrics.json` was written by
  `evaluate_and_save_metrics`, which scores through `predict_probabilities` (`train.py:122` →
  `train.py:104-111`) — a **single** forward pass. The document knows this; it is the defect item
  7 bullet 1 describes. So the one row in the table that *is* traceable to a repo artifact is
  methodologically mislabelled, and mislabelled along exactly the axis item 7 bullet 1 is about.
  Separately, row E's support is 520 real / 480 fake (`metrics.json:20, 26`) — unbalanced, while
  rows A-D are balanced 50/50. The "Images 1000" cell hides that, and it makes row E's accuracy
  not directly comparable to the rest.
* **Rationale & Recommendation:** Add "single pass, 520/480" to row E, and restate the header
  convention so it cannot be silently inherited.

### NM4. The README reconciliation misdiagnoses the conflict it is correcting, and the regeneration list omits a second in-repo document — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both, with different sub-points; merged)
* **Affected Scope:** lines 31-34, 345-346
* **Discrepancy Description:** Lines 31-32 say "`README.md` currently quotes row D's recall and
  specificity beside row A's accuracy." Half right. `README.md:19` reads "~89% accuracy on
  held-out images (80% AI recall, 86% real-photo specificity at threshold 0.55)": the recall and
  specificity do match row D (80.2/86.2), but the accuracy does not match row A (88.0%) — "~89%"
  matches **row E** (88.8%, *internal train/test split*, threshold 0.50) or
  `thresholds.json:6`'s 0.893. And `README.md:69` explicitly attributes it to the 1,600
  held-out images — the set this very table pins at **83.2%**. So the actual error is worse than
  the document says: an internal-split number, drawn from the training corpus, is being
  presented as a held-out figure, and the held-out figure for the set it cites is six points
  lower. "Both numbers are real but they come from different sets" understates it to the point
  of being reassuring. Three further README conflicts are not listed for regeneration:
  `README.md:69`'s 1,600 denominator (2,599 files on disk), `README.md:74`'s "~73%" hard-mined
  (row B is 75.5%, matching no row), and `README.md:83-84`'s "100% of the real-photo holdout
  set are square 1024×1024", which the prior review flagged as a 150-image sample stated as a
  property of 1,000 files.
  **And `REPORT.md` is tracked, carries stale numbers, and is mentioned nowhere in the plan:**
  `REPORT.md:73` states "24,763 real images and 24,053 fake images, giving a total of 48,816"
  against the plan's 23,944 / 48,707 (file counts support the plan); `REPORT.md:159-164`
  re-publishes 88.80% / 89.30% with no threshold or dataset attribution, i.e. row E and
  `thresholds.json` mixed; and `REPORT.md:116` still documents "Default Scan" and "AI-Sensitive
  Scan" modes that `README.md:17` says no longer exist and that no code implements (`main.py`
  has one `/predict` path, `inference.py:65-95` one scoring path; the `inference.py:72-76`
  comment confirms the mode was removed).
* **Rationale & Recommendation:** A plan whose headline structural change is "pinned all
  headline numbers to one table after finding five mutually inconsistent accuracy figures"
  must name every file that will carry stale ones. Fix the attribution at 31-34, list all four
  README conflicts, and add `REPORT.md` to item 1's regeneration scope — or delete its stale
  sections outright, which is cheaper.

### NM5. "2,257 holdout images" is wrong — there are 2,599 — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both); re-verified by the synthesizer
* **Affected Scope:** line 73
* **Discrepancy Description:** Counted: `data/holdout_newdb/real` = 1,000;
  `data/holdout_newdb/fake` = 999; `data/holdout_hard_mined/real` = 300;
  `data/holdout_hard_mined/fake` = 300. Total **2,599**, all `.jpg`/`.png`. The document is low
  by 342 files (13%). The figure was carried over from revision 1 unchecked — in the item the
  document calls "do this first", in a bolded sentence, in a revision whose own summary opens
  "all verified against the code" (line 327). It is also now the **third** mutually
  inconsistent holdout denominator in the project (1,600 in `README.md:69`, 2,257 here, 2,599
  on disk), in the revision whose stated purpose was to stop exactly that.
* **Rationale & Recommendation:** Correct to 2,599 and state the per-folder split, since item
  1's acceptance test needs it anyway (NB4).

### NM6. Item 4's prescribed augmentation cannot be written where item 4 points — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both); re-verified by the synthesizer
* **Affected Scope:** lines 197-200 vs `train.py:47-62`
* **Discrepancy Description:** The instruction is "add Gaussian noise to the augmenter in
  `train.py:47`, sampling σ per example across the measured attack range… and applying it to a
  fraction of each batch". `train.py:47-54` is a `tf.keras.Sequential` of stock preprocessing
  layers, and `tf.keras.layers.GaussianNoise` takes a **scalar** `stddev` at construction and
  applies the same σ to everything. Per-example σ and a Bernoulli batch mask are not expressible
  as a stock layer there. **This is also *why* revision 1 wrote a constant — the stock layer
  only accepts one.** Revision 2 fixed the constant's units and kept pointing at the block where
  the new behaviour is unavailable. The author will reach this line and stop. The fraction is
  also unspecified: revision 1 said ~20%, revision 2 dropped the number without replacing it.
* **Rationale & Recommendation:** The cheap answer already exists two lines down: the batch-level
  `train_ds.map(lambda images, labels: (augmenter(images, training=True), labels))` at
  `train.py:59-62` is the natural place for `tf.random.normal` with a per-example stddev from
  `tf.random.uniform([batch], 0.0, 0.125)` plus a keep-mask. One clause pointing at the map
  instead of the `Sequential` fixes it without reintroducing the pseudocode that prior N2
  correctly removed. State the fraction.

### NM7. Item 3a's three options are not interchangeable, and one of them silently inverts the unit convention item 4 depends on — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** lines 144-145
* **Discrepancy Description:** "Move to `tf.data` with map-decode, or
  `tf.keras.utils.image_dataset_from_directory`, or pin a fixed subset" are offered as
  equivalents. They are not. `image_dataset_from_directory` yields raw 0-255 float tensors and
  does **not** apply `preprocess_input`; adopting it moves normalisation out of `data.py:26` and
  into the graph. That inverts the answer to the question revision 2 spent its longest section
  getting right: if preprocessing happens *after* augmentation, the correct σ for the augmenter
  becomes 4/8/16, not 0.031/0.063/0.125, and the whole table at 188-192 flips. It also changes
  serving, because `load_image_for_inference` (`data.py:59-64`) would then be the only remaining
  place that preprocesses — training/serving skew, exactly the bug class the document is trying
  to close. "Pin a fixed subset" is also not equivalent: it caps the corpus and reintroduces the
  stale-baseline problem of item 3c.
* **Rationale & Recommendation:** Either drop the `image_dataset_from_directory` option or state
  the consequence. Note that if NB1's resolution is adopted (preprocessing as the model's first
  layer), `image_dataset_from_directory` becomes the *correct* choice and the conflict inverts
  into an argument for it.

### NM8. The threshold-fit fix is scheduled last but affects every retrain scheduled before it — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** lines 275-278 (item 7) vs item 3c (166) and item 4
* **Discrepancy Description:** Item 3c runs the retrain twice (line 166) and item 4 retrains
  again; each of those calls `find_best_threshold` on single-pass validation scores at
  `train.py:219-221`. Fixing the fit afterwards in item 7 invalidates the threshold every
  earlier retrain wrote, and item 4's acceptance numbers were produced at the wrong threshold.
  This is a one-line change and it belongs before the first retrain, not in the "small fixes"
  bin at the end. Prior N4's renumbering fixed the old numbering/priority conflict but did not
  re-derive which of item 7's bullets have ordering constraints — two do (this one, and the
  `confidence` deletion, which is user-facing and independent of everything else).
* **Rationale & Recommendation:** Move it ahead of item 3, or resolve NM1 by deletion, which
  retires it. Mark the remaining item 7 bullets as order-independent so it is clear that only
  these two are not.

### NM9. Item 2's gate has no bar, 2a and 2b have no stated relationship, and the pinned table has no invalidation rule — Severity: 🟡 Major

* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both; merged)
* **Affected Scope:** lines 110-120, 132-133, 139; pinned section 11-34
* **Discrepancy Description:** Three gaps in one mechanism.
  * **No bar.** "If 2a or 2b recovers most of σ=16 recall, item 4 downgrades from a retrain to a
    preprocessing change" (132-133). "Most" is undefined; there is no clean-accuracy bar even
    though 2a's own text (115) says it costs clean accuracy; and "a preprocessing change" never
    says which file changes. Item 4 has exactly the right shape of criterion two pages later
    (206-207: recall above 70%, clean within X of baseline) and is not referenced.
  * **2a and 2b are alternatives, not a pair, and nothing says so.** 2a is "always score at
    σ=8"; 2b is "average σ=0 and σ=8". Composing them yields σ ∈ {8, 8}, i.e. 2a again — there
    is no composition worth having. The real trade is 2a (free, 2 passes, loses the clean score
    entirely) versus 2b (4 passes, keeps it). A reader will reasonably try to run both.
  * **No invalidation rule.** Running 2a breaks item 1's pinned row A, which is pinned at
    threshold 0.55 (98-99); 2a says so itself (115); 2b shifts the averaged AI score by about
    half the −0.288 mean shift and is given no such caveat. Items 2a, 2b, 3c and 4 each
    invalidate rows A-D, and only the README gets a regeneration instruction (31-34).
  * **Unnamed liability of both.** Injected noise is random, so the served verdict becomes
    **non-deterministic** — the same upload can cross the threshold and flip from Real to
    AI-generated on re-submission. For a demo UI that is a visible defect. Seeding the RNG from
    a hash of the image bytes fixes it in one line; the document does not raise it.
* **Rationale & Recommendation:** Reuse item 4's criterion verbatim for item 2 — same numbers,
  same instrument — and the conditional at line 139 becomes mechanically evaluable instead of a
  judgement call the author has to make twice. Add one sentence saying 2a and 2b are
  alternatives and why. Add one line to the pinned section: "Any item that changes
  preprocessing, weights or the threshold retires rows A-D; regenerate with item 1's script
  before quoting them again." Add the seeding note.

### NM10. Item 6c's "drop-in change to `build_model`" is false, for the same root cause as NB1 — Severity: 🟡 Major

* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** lines 263-264
* **Discrepancy Description:** EfficientNet and ConvNeXt in Keras do not take MobileNetV2's
  [-1, 1] input, and preprocessing in this project is applied at *load* time (`data.py:26`), not
  in the model. Swapping the backbone therefore requires editing `data.py`, which silently
  breaks the unit convention item 4 establishes 60 lines earlier, plus `load_image_for_inference`
  and therefore serving and `predict_paths`.
* **Rationale & Recommendation:** Call it what it is — a backbone swap plus a preprocessing
  migration — or adopt NB1's resolution first, after which it genuinely is drop-in, because each
  Keras backbone carries its own normalisation.

---

### N1. "Out of scope" both declines work and prescribes it — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both). Line 303 defines the section as
  things "explicitly declined"; lines 319-321 then prescribe a required ~60-line deletion, and
  305-308 make item 6b a precondition for a declined item. A reader cannot tell whether the
  threshold deletion is work to do or work not to do. Split into "Declined" and "Deliberate
  deletions" — a heading, not a rewrite.

### N2. Line 268 is garbled — Severity: 🟢 Nit
* **Lens:** both. "Run through one set of weights in that order, 6b is evaluated on a model
  trained to discard the signal it needs." The conditional is missing ("**If** you run both
  through one set of weights in that order, …"); as written it is an imperative telling the
  reader to do the thing the paragraph is warning against.

### N3. Line-reference drift — Severity: 🟢 Nit
* **Lens:** both. One genuine misattribution: the `best < default + 0.02` guard is at
  `calibrate_thresholds.py:75-78`, cited alongside `:80` (the clip). Four pointer-to-caller
  references, harmless but worth tightening: `train.py:219` (the fit is :220), `train.py:224`
  (the write is :148), `inference.py:84` (the rescale is :85-87), `inference.py:13` (the band
  is :14).

### N4. "≈29 GB" is a floor, not a peak — Severity: 🟢 Nit
* **Lens:** `egm-inventive-catalyst`. `data.py:30-56` builds a Python list of 48,707 arrays and
  then copies it into `np.array`, and `train.py:32-45` `train_test_split` copies again. The real
  peak is roughly **2×** the quoted figure. Strengthens item 3a rather than weakening it.

### N5. "roughly 6,700 images" is the corpus, not the training set — Severity: 🟢 Nit
* **Lens:** `egm-inventive-catalyst`. Line 163: 1,000 test support at 15% implies a *corpus* of
  ~6,670; the training split was 70% of that, ~4,670 (`train.py:32-45`). The "sevenfold" growth
  claim is correct for the corpus.

### N6. The σ table's "vs clean" column is undefined, and the column that proves the point is missing — Severity: 🟢 Nit
* **Lens:** `egm-inventive-catalyst`. Lines 174-179. The numbers check out as accuracy deltas
  against the σ=0 row (88.2−85.2 = 3.0, −5.7, −12.9 ✓, and the σ=0 row matches row C exactly ✓),
  but the trailing ", real" is unexplained in a table about a Real/AI classifier — if it means
  "statistically real", write "significant". More usefully: real-specificity, derivable from the
  two printed columns on a balanced 300+300 set, goes 88.4 → 90.4 → 92.7 → 94.9 as σ rises.
  Printing that third column makes the asymmetry argument self-evident and removes the need for
  the mean-shift number entirely.

### N7. The mean-shift claim is unsupported in code, unpinned, and rounded generously — Severity: 🟢 Nit
* **Lens:** `egm-inventive-catalyst`. Lines 181-182 (−0.288 vs −0.081). Its only in-repo
  corroboration is `README.md:81` ("-0.29 vs -0.08") — a file the plan declares unreliable and
  plans to regenerate. No σ and no dataset are stated (presumably σ=16 on row C). The ratio is
  3.56, so "nearly four times" is generous; "3.6×" is accurate and costs nothing. Note the
  structural point: this is the one quantitative claim the pinned-numbers rule does not cover,
  because it is not an accuracy figure — so the rule has a hole exactly where the document's
  most-quoted *mechanism* claim lives.

### N8. The `confidence` deletion and digit fix touch more sites than stated — Severity: 🟢 Nit
* **Lens:** both. `index.html:369`'s `formatPercent` is consumed at 396, 397, 418, 419, 509 and
  510, and `"0.00%"` is hardcoded at 274 and 436. Removing the field also touches `main.py:68`
  (API payload), the documented response shape at `README.md:60`, and `predict.py:21`. A
  four-file change, not a one-file change. Worth one clause so the author does not discover it
  mid-edit.

### N9. `--threshold T` contradicts "prints at the live threshold" — Severity: 🟢 Nit
* **Lens:** `egm-inventive-catalyst`. Lines 79-82. Trivial, but the acceptance test depends on
  which wins.

### N10. "One `assert` at the bottom" is not the test item 7 asks for — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`. Lines 95-96 vs 290-292. Item 7 wants a `test_inference.py` that
  catches a preprocessing regression. An assert inside `eval_holdout.py` runs only when someone
  deliberately runs a multi-minute evaluation that needs the trained model and `data/` — and
  `data/` is gitignored (`.gitignore:11`). That is an acceptance run with an assert in it, not a
  smoke test. The two-image test is about ten lines and genuinely free; keep it separate.

### N11. `artifacts/` is gitignored, so "committed file" is the wrong word and the truth is worse — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`. `.gitignore:7-10` ignores `artifacts/` and `artifacts/*` and
  un-ignores only `!artifacts/ai_image_detector.keras`, so `metrics.json`, `thresholds.json` and
  `metrics_before_retrain_20260326_150355.json` are untracked. Lines 157 and 329 should say "the
  on-disk file", not "the committed file" — and more importantly, **there is no git copy to
  recover `thresholds.json` from** after a retrain overwrites it, which strengthens NB3. Item 3b
  should name a snapshot destination.

### N12. The pinned table's *inputs* are outside version control too — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`. Item 1's argument is that the scripts "are not in version control
  and will not survive" (68-70). `data/` is gitignored, so `data/holdout_newdb` and
  `data/holdout_hard_mined` will not survive either, and the regenerated table is reproducible
  only on this machine. For a 48,707-image student project that is the right call, but say it,
  because otherwise "one script is the sole producer of this table" (33-34) reads as
  reproducibility it cannot deliver. A `data/holdout_*/MANIFEST.txt` of filenames plus the seed
  is the cheap half-measure.

### N13. "No rotation is needed on repository grounds" will be read as "no rotation needed" — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`. Lines 294-296. The hedge is technically correct (see §4). But
  `artifacts/gemini_prompt_bank_500.txt`, `artifacts/gemini_prompt_bank_500.csv` and
  `artifacts/gemini_prompting_playbook.md` all exist, which is evidence the key was used against
  a live paid API from this machine. Deleting the working copy of a *live* key is a different
  decision from deleting a dead one, and "Revision 1 overstated this" nudges the reader toward
  doing nothing. One clause: "rotate if the key is still live — that is an account question, not
  a repository one."

### N14. Item 5's collection has human lead time and is scheduled after the item it would inform — Severity: 🟢 Nit
* **Lens:** `egm-zoom-out`. Collecting 100-200 AI images from an unseen generator is hours of
  *human* time, not compute, and line 226 says this is "where 88% is most likely to collapse".
  Item 4's retrain is validated only on row C — images from the same homogeneous corpus the model
  already knows. Starting the collection at the top of the plan costs nothing against the compute
  path and would give item 4 a second, far more informative acceptance set. A judgement call
  rather than an error, but it is the one reordering the renumbering missed.

---

## 3. Actionability Verdict

**No — the document is not yet actionable without further clarification.** Both lenses
independently produced near-identical lists of stop points. Consolidated, in execution order:

| # | Where the author stops | Finding |
| --- | --- | --- |
| 1 | Item 1 CLI shape — is `<folder>` the parent holding `real/`+`fake/`, or one class folder plus a label flag? Labels come from nowhere. (`calibrate_thresholds.py:98-99` shows the existing convention.) | NB4 |
| 2 | Where `--noise`/`--jpeg`/`--scale` attach, given `load_image_for_inference` decodes, resizes and preprocesses in one call with no seam | NB4 |
| 3 | Whether σ goes before or after the 224 resize — changes effective σ by ~4× and decides whether item 4's acceptance test measures the attack it names | NB4 |
| 4 | Which 300 images row A is, and whether "reproduces row A" means over HTTP or offline | NB4, NM2 |
| 5 | Whether to run 2a or 2b or both — composing them is a no-op | NM9 |
| 6 | What "recovers most of σ=16 recall" means numerically; and if 2a wins, which file changes and what re-fits the threshold | NM9, NM1 |
| 7 | Pick one of item 3a's three non-equivalent options, and resolve what it does to the unit convention | NM7 |
| 8 | Whether 3b's snapshot runs at all (line 139), and to what destination | NB3, N11 |
| 9 | What API gives per-example σ and fraction-of-batch, since `GaussianNoise` in a `Sequential` cannot — and what fraction | NM6 |
| 10 | "Clean accuracy within 1.5 points" of what, measured how — the same document calls ±2-3.5 points noise | NB5 |
| 11 | What exactly to check in `RandomContrast`/`RandomZoom` — the description does not match the real fault in one and there is no fault in the other | NB1 |
| 12 | Fix the threshold fit or delete the fitter | NM1 |
| 13 | Is "Out of scope" declined work or required work | N1 |
| 14 | Whether rows A-D survive 2a / 3c / 4 — no invalidation rule | NM9 |
| 15 | Delete `confidence` or not — line 287 explicitly defers the decision without naming a criterion | M9 residue |
| 16 | Which fake-class corpus to retrain on, as-is or with the 7,903 PNGs re-encoded | NB2 |

**Ordering verdict.** The renumbering is a genuine improvement: 1 (measure) → 2 (free
experiments) → 3 (prerequisites) → 4 (retrain) is the right spine, the free-before-expensive
inversion is correct, and revision 1's two-way numbering conflict is gone. But renumbering
introduced three new sequencing problems, in descending severity: item 3b's protection is now
conditional on the wrong branch (**NB3** — loss of the one artifact no code can regenerate);
item 7's threshold-fit fix sits after every retrain it affects (**NM8**); and item 5's
collection, which costs human lead time rather than compute, sits after the item whose
acceptance it would strengthen (**N14**). Fixing all three is pure reordering — no new content,
no new code.

**Scope verdict against the standard.** Still well-judged for a student mini-project, and
clearly better than revision 1. §2 (Simplicity/YAGNI) and §4 (Leverage Existing) now pass on
substance: the bootstrap machinery became a deletion, Platt scaling and the FFT branch are
declined with reasons, the cheap spectral probe replaced the expensive branch, and three real
repo utilities are named for reuse. The remaining failures sit in §0 (Context Sufficiency) and
§3 (Right Placement).

---

## 4. Assessment of the Author's Pushback on N1

**The pushback is correct, and prior N1's recommendation was overstated.**

Prior N1 said "rotation stays mandatory regardless, since deleting the working copy does not
remove it from history." That reasoning is sound *only if* the file was ever committed. The
author states they verified the file is untracked **and** absent from all git history. If that
holds, there is no history to purge, and deletion does suffice **on repository grounds** —
which is the only ground a repository review has standing to assert. Revision 2's wording at
lines 294-296 ("no rotation is needed on repository grounds… Revision 1 overstated this") is
the correct narrow claim, correctly hedged.

Two caveats, neither of which rescues the original finding:

1. **Neither reviewer could independently verify the history claim.** Both lenses are
   read-only and have no git access. `.gitignore:6` is confirmed, the file is confirmed present
   on disk, and no `*.py` in the project references `gemini|genai|GEMINI`. The "absent from
   every commit" half rests entirely on the author's own check. That is reasonable to accept —
   the author is the one who can run `git log --all -- gemini_key.txt` — but the word "verified"
   at line 295 is the author's verification, not the review's, and the document should not
   present it as code-verified alongside the other three corrections in the same list.

2. **Rotation may still be warranted on *account* grounds, which is a different argument.**
   `artifacts/gemini_prompt_bank_500.txt`, `artifacts/gemini_prompt_bank_500.csv` and
   `artifacts/gemini_prompting_playbook.md` all exist, which indicates the key was used against
   a live paid API from this machine. A live key sitting in a plaintext file on a student laptop
   is worth rotating for reasons that have nothing to do with git. The document's framing
   ("Revision 1 overstated this") risks being read as "no rotation needed", full stop. One
   clause closes it: *"rotate if the key is still live; that is an account question, not a
   repository one."*

**Net:** prior N1's "mandatory regardless" is withdrawn. The finding is correctly resolved.
Keep the narrow wording and add the account-grounds clause (N13).

---

## 5. Consolidated Change List for Revision 3

Ordered by payoff per unit of work. Items 1-4 are reordering or single sentences.

1. **Pull item 3b out of item 3 and make it unconditional** (NB3). One cut-and-paste.
2. **Decide the threshold subsystem once** — delete the fitter, freeze 0.55, add
   `--fit-threshold` to item 1's script (NM1, NM8). Net deletion.
3. **State the paired-comparison convention at line 26** and correct the per-row CIs (NB5).
   One sentence makes item 4's gate decidable.
4. **Correct line 73 to 2,599**, label row E "single pass, 520/480", and fix the README
   attribution at 31-34 (NM5, NM3, NM4).
5. **Repoint item 4's change at `train.py:59-62`** instead of the `Sequential` at :47, and
   state the batch fraction (NM6).
6. **Specify item 1's seam and injection point**, adopt `balanced_sample` with
   `--per-class/--seed/--filter`, and change the acceptance test to exact-match-on-rerun
   against row C (NB4, NM2).
7. **Add the PNG/JPEG confound** as a free measurement in item 2 and a corpus fix before item
   3c's re-baseline, with the "accuracy will drop and that is a correction" warning stated in
   advance (NB2).
8. **Fix `RandomContrast`** — either `value_range=(-1, 1)` as a one-liner, or move
   `preprocess_input` into `build_model` and retire four pieces of unit bookkeeping at once
   (NB1, and it also resolves NM7 and NM10).
9. Add item 2's numeric gate, the 2a-vs-2b relationship, the table invalidation rule and the
   noise-seeding note (NM9).
10. Add `REPORT.md` to the regeneration scope or delete its stale sections (NM4).
11. Sweep the nits: N1-N14.
