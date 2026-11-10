# Goldfish Review Report — Stage 7 (Critic Review)

**Target:** `docs/architecture-improvement-plan.md` (revision 4, full file, 558 lines)
**Range:** FeatureRange = whole document. DetailedRange N/A (no implementation-plan section); Stage 9 will not run.
**Date:** 2026-11-10

## Reviewer Status

| Reviewer | Standard | Status |
| --- | --- | --- |
| `egm-design` | `resources/document_standards/feature-spec-standard.md` | completed |
| `egm-structural` | `resources/document_standards/feature-spec-standard.md` | completed |
| `egm-style` | none | completed |

All three explored the repository. Severity is calibrated for a student mini-project:
production-process demands are not Blockers here. A Blocker means **an executor
following this document cannot complete the next action, or will complete it wrongly
and not find out**.

## Severity counts

| Severity | Count |
| --- | --- |
| 🔴 Blocker | 3 |
| 🟡 Major | 26 |
| 🟢 Nit | 18 entries (several are clusters) |

## Items 0 and 1: verification result

Both DONE claims are **true at the code level**, verified independently by two lenses.

- **Item 0.** `train.py:56` now reads `RandomContrast(0.1, value_range=(-1, 1))`, with the
  mechanism documented in a comment at `train.py:52-55`. `data.py:26` does apply
  `preprocess_input` at load time, so the hazard was real and the fix is correct.
  `RandomRotation` (`train.py:50`) and `RandomZoom` (`train.py:51`) genuinely lack the fault.
- **Item 1.** `eval_holdout.py` exists and does what the document says: reuses
  `iter_image_files` (`:32`, `:69`), `preprocess_image` (`:108`), `load_calibration`
  (`:161`); flip-averages at `:90` (`reshape(-1, 2).mean(1)` over sigmoids), matching
  `inference.py:77-78`; applies perturbations at `:101-106`, before `preprocess_image`
  and so before the 224 resize at `data.py:23`; `--check` gates accuracy and
  `--check-recall` gates AI-recall, either failing exits 1 (`:194-208`). The CI helper
  at `:124-128` reproduces the document's ±2.6 and ±5.6.

The defects below are not about whether items 0 and 1 landed. They are about what the
document now claims those two items enable (B-1, M8, M11) and what it still asks the
reader to do about them (M14, M21, M23).

---

# 🔴 Blockers

## B1 — Item 2's decision rule cannot be run with the committed harness
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** lines 208-219 (decision rule), 186-197 (2a, 2b)
* **Prior status:** `zoom_out_report_rev2.md` NB4 bullet 3 — **UNRESOLVED.** Revision 4
  closed the "no number" half and left the instrument half open.
* **Discrepancy:** The rule gives two commands,
  `eval_holdout.py data/holdout_newdb --noise 16 --check-recall 0.70` and `... --check 0.85`.
  Neither invokes 2a or 2b, and neither can. **2a is two noise stages** (a fixed σ≈8
  normalisation grain on every input, plus the σ=16 attack); `eval_holdout.py:105-106`
  calls `apply_noise` exactly once, with `args.noise`, and there is no normalisation-stage
  flag. **2b is a change to the scoring function**; `score()` (`eval_holdout.py:78-121`)
  stacks exactly `prepared` and its flip, with no noise-TTA mode. The document's own
  like-for-like comparison therefore measures the unmodified system on both branches and
  reports the current ~55% twice. The author runs both commands, sees no change, and
  cannot tell whether 2a or 2b work.
* **Recommendation:** add two flags to `eval_holdout.py` before item 2 —
  `--normalise-noise S` (applied after the attack, inside `score`, before
  `preprocess_image`) and `--noise-tta S` (stack four variants instead of two in `flush`).
  Roughly 15 lines, reusing `apply_noise` and the existing batch-pair reshape. Then the two
  commands are honest. This is the plan's declared *next action*, which is why it ranks
  above everything else here.

## B2 — Items 3b and 3b-bis are gated behind the branch that does not need them; the cheap path destroys the one artifact no code can regenerate
* **Reviewer Lens:** `egm-design` (B-2) and `egm-structural` (S1) — converged independently
* **Affected Scope:** line 234 ("Only needed if item 2 does not dissolve item 4") vs
  190-191, 222-224, 244-266
* **Prior status:** `zoom_out_report_rev2.md` NB3 — **UNRESOLVED and unacknowledged.**
  Raised as a Blocker against revision 2, whose own change list called the fix "one
  cut-and-paste". Revisions 3 and 4 neither did it nor mentioned it.
* **Discrepancy:** Verified end to end. (1) 2a and 2b both invalidate the 0.55 threshold;
  the document says so twice (190-191, 222-224) and instructs a re-measure. (2) The only
  committed re-measurement tool is `calibrate_thresholds.py`, whose `--output` defaults to
  `THRESHOLD_PATH` (`:95`), so it **overwrites `artifacts/thresholds.json`**; and
  `find_best_threshold` (`:81-88`) never emits `tuning_note` — the note at
  `thresholds.json:10` is the sole justification for freezing 0.55 and is exactly what the
  out-of-scope section tells the author to cite. (3) `artifacts/` is untracked
  (`.gitignore:9`), so there is no git copy. On the branch the plan is *hoping for*, the
  protection step is skipped and the irreplaceable artifact is lost.

  The design lens adds a second, worse layer: the re-measure is not only destructive, it is
  **not possible to do correctly**. `calibrate_thresholds.py:98-99` samples from
  `PROCESSED_DATA_DIR` (the *training* corpus, not the holdout), and `predict_paths`
  (`:30-52`) calls `load_image_for_inference`, which decodes, resizes and preprocesses in
  one call with no seam for 2a's normalisation grain. Revision 2 (NM1) recommended adding
  `--fit-threshold` to item 1's script; committed `eval_holdout.py` has only `--threshold`
  as an override (`:141-142`). **No instrument in the repository can fit a threshold for the
  2a/2b pipeline.** The cheap path has an undischarged hard dependency.

  Separately, **3b-bis is not a retrain prerequisite at all.** It is a live deployment
  correctness defect — the running Space uses `inference.py:13`'s constant, matching the
  local artifact only by coincidence (`.gitignore:9-10` + `Dockerfile:20` +
  `inference.py:34-45` confirm) — with zero dependency on retraining, filed inside a
  section gated on retraining.
* **Recommendation:** promote 3b and 3b-bis out of item 3 into their own unconditional
  section executed **before item 2**, i.e. before anything writes to `artifacts/`
  (structural: "pure relocation, no new content; renumber downstream, leave 3a and 3c behind
  the conditional"). Name the snapshot destination —
  `artifacts/metrics_before_retrain_<ts>.json` is the existing convention. In the same pass
  either add `--fit-threshold` to `eval_holdout.py`, or state that 2a/2b are measured at a
  hand-pinned `--threshold` and the re-fit is deferred.

## B3 — Item 3a's three options are not interchangeable, and one silently inverts the unit convention item 4 is built on
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** lines 240-243
* **Prior status:** `zoom_out_report_rev2.md` NM7 — **UNRESOLVED.**
* **Discrepancy:** "Move to `tf.data` with map-decode, or
  `tf.keras.utils.image_dataset_from_directory`, or pin a fixed subset" are still offered as
  equivalents. `image_dataset_from_directory` yields raw 0-255 tensors and does **not**
  apply `preprocess_input`. Adopting it moves normalisation out of `data.py:26`, after which
  item 4's conversion table (297-301) is **inverted**: the correct `stddev` becomes 4/8/16,
  not 0.031/0.063/0.125. An author who picks option 2 and then follows item 4 literally
  trains with noise ~127× too weak — the exact error class revision 2 exists to have
  corrected — and finds out only after a multi-day CPU retrain on 48,707 images. It also
  leaves `load_image_for_inference` (`data.py:59-64`) as the only remaining preprocessor,
  producing train/serve skew. "Pin a fixed subset" is also not equivalent: it caps the
  corpus and reintroduces the stale-baseline problem 3c exists to remove.
* **Recommendation:** delete the `image_dataset_from_directory` option, or attach one
  sentence: "if this option is taken, item 4's σ values are 4/8/16, not the preprocessed
  values, and `data.py:26` must move into `build_model`." Mark "pin a subset" as a fallback
  that forfeits 3c. The style lens asks for the same fix from its own angle: at minimum mark
  the default, since three options with no selection criterion means the reader picks by
  coin flip.

---

# 🟡 Major

## Sequencing, dependencies and payoff — `egm-design`

### M1 — On the cheap path item 0's fix is never cashed in, and the plan never counts that as a cost
* **Scope:** lines 55-64, 179-184, 232-236
* The document correctly states that rows A-D predate item 0's fix and that the deployed
  weights were trained while 54.5% of every augmented image was blanked. It then frames the
  retrain purely as the most expensive thing in the plan, with item 2 as the hoped-for
  escape. But the committed `RandomContrast` fix **changes nothing until a retrain happens**.
  If item 2 succeeds, the plan's own most dramatic finding is a no-op, the deployed model
  remains the one trained on mutilated data, and nobody ever measures what a correctly
  trained model scores. This is a cost/benefit inversion on the document's own evidence: the
  largest *unquantified* upside anywhere in the plan is "train without destroying half of
  every augmented image", and the preferred branch forecloses it. Item 2 does not dissolve
  "the retrain"; it dissolves only the *noise-augmentation* retrain.
* **Recommendation:** split the retrain in two. 3c's no-noise arm becomes unconditional,
  justified by item 0 alone, producing the post-fix baseline for everything downstream. Item
  2's success then cancels only the *second* arm. State plainly: "item 0's fix delivers
  nothing until 3c runs."

### M2 — Items 3c and 4 are circularly ordered
* **Scope:** line 276 ("The retrain runs twice: once without noise augmentation, once with")
  vs item 4's "Change" at 304-309
* 3c's second run *is* item 4's trained model. It cannot be executed before item 4's
  augmentation code exists, yet item 3 is declared a blocking prerequisite of item 4. As
  written the order 3 → 4 is unexecutable.
* **Recommendation:** restate as 3a/3b/3b-bis → write item 4's augmentation → 3c run A (no
  noise, produces the baseline) → 3c run B (with noise, *is* item 4) → item 4's two gates.
  One sentence, no content change.

### M3 — Every acceptance command omits `--threshold`, and every path in the plan moves the threshold
* **Scope:** lines 211-214, 317-320
* `eval_holdout.py:161` reads the live threshold from `thresholds.json` unless `--threshold`
  is passed. For **item 2**: 2a and 2b invalidate 0.55 by the document's own admission, but
  `--check 0.85` is a floor derived from row C measured *at* 0.55, so the comparison spans
  two different decision rules. For **item 4**: any retrain calls `find_best_threshold` and
  writes the result, and because the `best < default + 0.02` guard
  (`calibrate_thresholds.py:75-78`) demonstrably fired for the current artifact
  (`thresholds.json:5-6` show `validation_accuracy == validation_accuracy_default_0_5 ==
  0.893`, which only occurs on the guard branch, and that branch forces
  `best_threshold = 0.5`), **the next retrain writes 0.50, not 0.55**. Item 4's acceptance
  then runs at 0.50 while the 70% recall bar was set against a 55.7% measurement at 0.55.
  Lowering the threshold mechanically raises AI-recall under noise — worth several points
  given the measured −0.288 mean shift — so gate 1 mixes a threshold move into what is
  presented as a robustness measurement. A marginal pass would be uninterpretable.
* **Recommendation:** add `--threshold 0.55` to all four acceptance commands and say why:
  the gates measure a change in the *model*, so the decision rule must be held fixed. One
  flag per command, already implemented.

### M4 — The training σ range and the measured attack σ sit at different points in the pipeline; `/127.5` is not the whole conversion
* **Scope:** lines 293-309
* **Prior status:** NB4 bullet 2 — **UNRESOLVED.** Revision 3 documented the injection point
  in `eval_holdout.py:15-18` but never reconciled it with item 4's training range.
* All measured σ values (the table at 283-288, and `eval_holdout.py:101-106`) are applied to
  the decoded full-resolution image **before** the 1024→224 resize at `data.py:23`. Item 4's
  prescribed training noise is applied by the augmenter at `train.py:63-66`, which operates
  on **already-resized 224 tensors**. `cv2.resize` with default `INTER_LINEAR` combines
  roughly 2×2 source taps, so i.i.d. noise is attenuated by ~1.3-2× on the way in: a σ=16
  pre-resize attack reaches the network as roughly σ=8-12 equivalent. "Sampling σ per example
  across the measured attack range (0 to 0.125 preprocessed)" therefore over-shoots the
  attack it names by ~1.5-2×. Not fatal for gate 1 — it errs toward robustness — but the
  surplus is paid directly out of **gate 2**, for robustness gate 1 never tests. The document
  presents `σ_preprocessed = σ_pixel / 127.5` as the complete conversion, and that identity
  does not hold across the resize.
* **Recommendation:** state the pre-resize/post-resize distinction once, and either set the
  training range to ~0-0.08 preprocessed, or keep 0.125 and justify it as deliberate
  over-coverage with the cost charged to gate 2's budget. Measuring the attenuation is a
  five-line check with existing pieces. The style lens asks for the same thing notationally:
  write `σ_upload` and `σ_train`, or carry "(pre-resize)" / "(post-resize, preprocessed)" on
  every σ.

### M5 — Item 4's change is pointed at `train.py:47`, where the prescribed behaviour is not expressible
* **Scope:** line 306
* **Prior status:** NM6 — **UNRESOLVED across two revisions.**
* `train.py:47-58` is a `tf.keras.Sequential` of stock preprocessing layers.
  `tf.keras.layers.GaussianNoise` takes a **scalar** `stddev` at construction and applies it
  uniformly. "Sampling σ per example" and "applying it to a fraction of each batch" are not
  expressible there — which is precisely why revision 1 wrote a constant. The author reaches
  this instruction and stops. The natural location is four lines down, the batch map at
  `train.py:63-66`, where `tf.random.normal` with per-example `stddev` from
  `tf.random.uniform([batch], 0.0, σ_max)` plus a Bernoulli keep-mask is straightforward.
  The batch fraction is also still unspecified (revision 1 said ~20%, revision 2 dropped it,
  3-4 did not replace it) — and it is the single parameter that trades gate 1 against gate 2.
* **Recommendation:** repoint the sentence at `train.py:63-66` and name the fraction.

### M6 — Item 4 has no failure branch, and gate 2's satisfiability is asserted, not evidenced
* **Scope:** lines 314-333
* The document's risk coverage is excellent for the *current* system and nearly absent for
  the *proposed* changes. Item 4 is the plan's most expensive item and presents two gates
  with no statement of what happens if gate 1 passes and gate 2 fails — the most likely
  outcome of noise augmentation, and the trade-off the document itself names ("costs some
  clean accuracy", line 190). Revision 1 had "halve the σ and retry"; revision 4 has nothing.
  No evidence anywhere shows σ=16 recall ≥70% and clean accuracy within 3 points are
  *jointly* achievable; the only relevant datum (σ=8 recall is 72.3% on the un-augmented
  model) says nothing about the post-augmentation trade curve.
* **Recommendation:** three lines. (a) Gate 1 passes, gate 2 fails → reduce the batch
  fraction (not σ_max) and re-run; the fraction is the clean/robust dial. (b) Both fail → the
  retrain is a dead end and item 2's best number stands as the shipped answer. (c) Record the
  Pareto point actually achieved rather than pass/fail, since a 65%-recall / 0-point-
  regression model may be the better product than a 72% / −3 one.

### M7 — The threshold decision must be made before item 3c, the deletion branch has two unstated consequences, and the decision lives in two places, one of them headed "declined"
* **Reviewer Lens:** `egm-design` (M-7) + `egm-structural` (S11)
* **Scope:** lines 424-433 vs 268-276 and 456-480
* **Prior status:** the *contradiction* (NM1 / Stage 6 #4) is genuinely resolved — the entries
  are now framed as alternatives with a stated preference. **NM8, the ordering half, is
  UNRESOLVED.** `zoom_out_report_rev2.md` **N1** (out-of-scope section both declines and
  prescribes work) is also **UNRESOLVED.**
* **Design view:** `train.py:224-225` calls `find_best_threshold` and overwrites
  `thresholds.json` on **every** retrain. Item 3c runs the retrain twice and item 4 runs it
  again. Leaving the decision until item 7 ("small fixes", last) means three retrains write a
  threshold the plan intends to delete or repair, and item 4's gates are measured at it (M3).
  Two consequences of the preferred deletion are unstated: (1) `find_best_threshold` is also
  the **only** producer of `uncertain_low` / `uncertain_high`
  (`calibrate_thresholds.py:83-84`), which `inference.py:44-45` reads to build the
  `low_confidence` band — so "freeze the threshold" must be "freeze the threshold **and the
  band**"; (2) after the deletion **nothing** writes `thresholds.json`, and the file is
  untracked, so a fresh clone or Docker build falls back silently to `inference.py:13`. The
  deletion branch is only safe if 3b-bis is resolved first, in the "track the artifact"
  direction. The document handles these in two separate sections and never links them.
* **Structural view:** both passages are *full* specifications of one decision (430-431 and
  474-480), and a required ~60-line deletion is filed under a heading defined at line 458 as
  "explicitly declined rather than quietly dropped".
* **Recommendation:** one home for the decision, ahead of item 3. Either split "Out of scope"
  into "Declined" and "Deliberate deletions" and put this in the latter with its ordering
  constraint stated, or make it a short "Decide before item 3" block near line 234 with
  one-line pointers left in item 7 and Out of scope. Take the deletion and bind it to 3b-bis:
  add `!artifacts/thresholds.json` to `.gitignore`, commit the frozen
  `{threshold, uncertain_low, uncertain_high, tuning_note}`, delete `find_best_threshold` and
  `train.py:12, 224-225`, and make `load_calibration` raise instead of defaulting.

### M8 — "One script is the sole producer of this table" is false for four of five rows, and rows A, B and D are orphaned
* **Reviewer Lens:** `egm-design` (M-8) + `egm-structural` (S8)
* **Scope:** lines 38-44, 67-69, 173-175
* **Prior status:** NM2 — **substantively UNRESOLVED.** Revision 4 addressed the pre-item-0
  caveat (58-64); the sole-producer and orphaned-rows halves are open.
* **Design view:** `eval_holdout.py` takes a single positional `folder` containing `real/` and
  `fake/` (`:135`, `:60-75`). **Row A** is "served over HTTP" — no HTTP mode exists and the
  300 images are an unrecorded subset. **Row B** is mined against *the model being evaluated*
  (filenames carry the mining scores); the glossary now explains why ~75% is expected, but
  never says row B **dies the moment the model is retrained** and is not comparable to its own
  former value afterwards. **Row D** merges two folders at an unstated ratio and the script
  cannot take two folders, so row D is not producible at all. **Row E** needs the old
  `train.py`. Only row C is regenerable — yet the stated residual task is "regenerate the
  pinned-numbers table and `README.md` from this script".
* **Structural view:** grepped every mention. No item gates on, regenerates, or compares
  against A, B or D. Row A appears only at 47 (its own CI) and 66; row D at 47 and 66; row B
  at 17, 47 and 48 — three of its four mentions are about its own error bar. Every executable
  gate in the document is anchored to row C or to 3c's future re-baseline.
* **Recommendation:** demote A, B and D to footnotes marked non-regenerable (row B marked
  "retires on retrain") and pin the full-set `holdout_newdb` number as the single headline.
  That makes item 1's residual task executable and removes a four-row maintenance burden.

### M9 — The README/REPORT reconciliation is misdiagnosed, under-scoped, and parked inside an item labelled DONE
* **Reviewer Lens:** `egm-design` (M-9) + `egm-structural` (S14) + `egm-style` (S9) — all three
* **Scope:** line 23 (glossary), 66-69, 173-175
* **Prior status:** NM4 — **UNRESOLVED** for the accuracy half; the `REPORT.md`-in-scope half
  is resolved (lines 23, 174).
* The document says `README.md` "quotes row D's recall and specificity beside row A's
  accuracy". Verified: `README.md:19` reads "~89% accuracy on held-out images (80% AI recall,
  86% real-photo specificity at threshold 0.55)". The recall/specificity do match row D
  (80.2 / 86.2), but "~89%" is **not** row A's 88.0% — it matches row E (88.8%, *internal
  train/test split*, threshold 0.50) or `thresholds.json:6`'s 0.893. And `README.md:69`
  attributes it explicitly to the 1,600 held-out images, which this table pins at **83.2%**.
  So the real defect is an internal-split number drawn from the training corpus presented as
  a held-out figure, six points above the held-out figure for the set it cites. The style lens
  notes that "Both numbers are real but they come from different sets" describes a bookkeeping
  slip where the error is a confidence-level error — revision 2 said this wording
  "understates it to the point of being reassuring", and it is unchanged.

  `REPORT.md` is worse than "older accuracy figures". `REPORT.md:27, 113-116, 147, 190-192`
  document **two scan modes** ("Default Scan" at threshold 0.65 with rotation-based
  conservative scoring; "AI-Sensitive Scan" at 0.40) and a three-way verdict including
  "uncertain". No code implements any of it — `main.py` has one `/predict` path,
  `inference.py:65-95` one scoring path, `README.md:17` says the modes are gone. The graded
  academic artifact describes a system that does not exist, and framing the fix as a numbers
  reconciliation will not catch it. Three further README conflicts are unlisted: the 1,600
  denominator (2,599 on disk), "~73%" hard-mined vs row B's 75.5%, and `README.md:83-84`'s
  "100% of the real-photo holdout set are square 1024x1024" (a 150-image sample stated as a
  property of 1,000 files).
* **Recommendation:** split the documentation reconciliation into its own numbered item with
  its own slot in the order, so it is schedulable rather than buried in a DONE item. Correct
  line 66's attribution with the row-E diagnosis, list all four README conflicts with line
  numbers (`:19`, `:69`, `:74`, `:83-84`) so the executor knows when the task is finished, and
  scope the `REPORT.md` work as "delete the two-scan-mode architecture description", not
  "update numbers".

### M10 — Item 6b's noise statistics reintroduce at serving the attack item 4 exists to close
* **Scope:** lines 395-418
* The *named* conflict (item 4 teaches the backbone to ignore high-frequency energy; 6b
  exploits it) is correctly conditionalised now, and the stated mitigation **does hold on its
  own terms**: because 6b computes its scalars on the raw pre-preprocessing image, item 4's
  augmentation cannot destroy them, and the incremental value measured against a
  noise-invariant CNN logit is if anything conservative. The unnamed half is the serving
  direction, and it is the dangerous one. 6b's features are "estimated noise σ,
  high-frequency band energy ratio, residual variance after a 3×3 median filter"; adding σ=16
  grain to an AI image moves **all three** toward real-photo values. A logistic regression
  leaning on them is defeated by exactly the attack item 4 spent a retrain closing — and 6b's
  own evaluation, on clean validation data, cannot detect that. The liability note
  ("hand-crafted noise statistics are easy to attack directly") gestures at this but never
  connects it to item 4, so a reader concludes the conflict is fully mitigated.
* **Recommendation:** add one gate — 6b may not be promoted into the served scorer until
  `eval_holdout.py --noise 16 --check-recall 0.70` passes on the **combined** scorer, not the
  CNN alone. State that a 6b improving clean accuracy while failing that gate is a regression.

### M11 — The PNG confound was retired on the weaker of two available tests, and the harness cannot run the stronger one (or item 2c)
* **Scope:** lines 92-100, 198-206
* **Prior status:** NB2 was a Blocker, demoted to a measured non-problem in revision 3. NB4
  bullet 4 (`--filter`) — **UNRESOLVED.**
* The demotion rests on an **unpaired between-subset** comparison (PNG fakes 0.858 vs JPEG
  fakes 0.848, CI [-0.049, +0.060]). That is the right first-order test and the conclusion is
  probably correct, but two weaknesses should not be glossed. (1) The far more powerful test
  is **paired** — re-encode the same 267 PNG fakes and measure the score shift on those
  images. The document invokes it ("the `--jpeg 30` run agrees … barely moves") but reports
  only whole-set accuracy (88.0% vs 88.3%), in which a 267-image effect is diluted ~7:1 and
  invisible; q30 is also far harsher than a real save path. (2) `eval_holdout.py` has **no
  `--filter`/glob flag** (`collect`, `:60-75`, takes a folder and an integer limit only), so
  neither the PNG-subset measurement nor **item 2c's 710-file phone-photo slice** can be run
  with the committed harness. Item 2c — one of the three experiments the plan runs next — is
  unexecutable for the same reason as B1.
* **Recommendation:** add `--filter GLOB` to `collect` (three lines). Report the paired
  mean-p(AI) shift on the 267 PNG fakes at `--jpeg 90`. If also null, the demotion is properly
  argued; attach the dataset-hygiene note to item 3 so it belongs to an item rather than
  floating in the non-problems section.

### M12 — Item 5's zero-dependency half sits at position 5, after the item it would most improve
* **Scope:** lines 337-376
* **Prior status:** N14 — **UNRESOLVED.**
* Item 5's own text says provenance is "where 88% is most likely to collapse", and its
  prerequisite — write down which generators are in `data/processed/fake` — is a desk task
  with no compute dependency, no code, and no prerequisite of its own. The collection is human
  lead time, not machine time, so it runs in parallel with the compute path for free.
  Meanwhile item 4's acceptance is measured entirely on row C: images from the same
  homogeneous corpus the model already knows. That is the payoff misjudgement — the plan's own
  highest-stated risk is scheduled behind items that cannot test it.
* **Recommendation:** move the provenance write-down to an unconditional slot alongside
  3b-bis, and start the 100-200 image collection at the top of the plan. Even 100
  unseen-generator images before item 4 completes give item 4 a second, far more informative
  acceptance set at zero schedule cost.

### M13 — Item 2c settles for a ceiling when a clean number is free on the retrain path
* **Scope:** lines 198-206
* Raised in revision 1 (M4) and again in revision 2; dropped both times without comment. Line
  206 defers the question ("group-split by scene if they are ever held out"). But on the
  retrain path a retrain is happening anyway — holding the 710 `phone_real_IMG_*` files out of
  `data/processed/real` during 3c costs zero collection and zero extra compute, and converts
  the project's only real-distribution number from an admitted ceiling into a clean
  measurement. Given M1 (the re-baseline retrain should be unconditional), this is available
  on every branch.
* **Recommendation:** one sentence in 3c — exclude `phone_real_IMG_*` from the training split
  for the re-baseline run, grouped by scene, and report 2c against it as a real number.

## Structure — `egm-structural`

### M14 — Item 1 is headed DONE but carries three open deliverables
* **Reviewer Lens:** `egm-structural` (S2) + `egm-style` (S1)
* **Scope:** line 7, heading at 125, lines 147-149, 173-175
* **Prior status:** `comprehension_report.md` finding 1 — **half resolved.** Revision 4 fixed
  the tense of the motivation paragraph; the DONE-with-leftovers structure remains, and
  revision 4 *added* line 7's "**Items 0 and 1 are done.**", which inherits the defect.
* Three pieces of live work sit under a DONE label: "**Still to do here:** regenerate the
  pinned-numbers table and `README.md` … and reconcile `REPORT.md`" (173-175), and "It is not
  the two-image sanity assert … that remains open" (148-149). Verified open: `README.md:19`
  and `:69` are unchanged. DONE status is discoverable from headings;
  *not-done-inside-DONE* is not. A reader who trusts line 7 skips two real tasks, and a
  reader asking "what do I do next" has to read a section flagged as finished.
* **Recommendation:** retitle line 125 "Item 1 — PARTLY DONE: harness committed, number
  regeneration open", or (preferred, and it composes with M9) lift 173-175 and 148-149 into
  their own numbered item in execution order. Rewrite line 7: "Item 0 is done. Item 1's script
  is committed; two tasks under it are open."

### M15 — Sub-items are the unit of cross-reference but are not headings
* **Scope:** bold leads at 186, 193, 198, 238, 244, 256, 268, 381, 392, 405; pointers at 36,
  56, 62, 120, 147, 184, 208, 273, 322, 326, 329, 342, 390, 415, 462, 477
* The document is cross-referenced at sub-item granularity ("item 3c", "3b-bis", "6b",
  "item 2c") roughly fifteen times, but 2a-2c, 3a/3b/3b-bis/3c and 6a-6c are bold paragraph
  leads, not headings. They appear in no table of contents, have no anchors, and `3b-bis` in
  particular is an identifier a reader must find by full-text scan of a 558-line file. Heading
  hierarchy is otherwise clean (H2 per item; `### Measured non-problems` correctly nested).
* **Recommendation:** promote each sub-item to `###` under its item. That alone makes every
  existing pointer navigable without rewriting any prose.

### M16 — Item 4's gate is written out twice, the copies diverge, and the divergence is a real 0.2-point loosening
* **Reviewer Lens:** `egm-structural` (S4, S7) + `egm-style` (S4)
* **Scope:** lines 208-219 vs 314-330; row C's literal value at 42, 155, 217, 285, 330
* Line 208 says "Judge 2a and 2b against item 4's own bar" — then re-prints the commands
  instead of pointing at them. Command 1 is identical in both places; command 2 is
  `--check 0.85` at 214 but `--check <re-baseline minus 0.03>` at 319, reconciled by a third
  sentence at 329-330. Three places to keep in sync for one gate.

  **The copies have already drifted, and the drift is numeric: 88.2 − 3 = 85.2, not 85.0.**
  The prose states one bar and the copy-pasteable command states another, 0.2 points looser —
  and a reader runs the command, not the prose. The root cause is structural: line 281 already
  says item 4's evidence was "measured on row C's images", then line 285 re-prints 88.2% /
  88.0% as the σ=0 row, and 217 and 330 repeat the literal "88.2%" instead of citing the row
  ID the Pinned numbers section exists to provide. The document's own rule at line 33 ("Every
  accuracy figure in this document refers to a row below") is undermined by its own body
  copying values out.
* **Recommendation:** define the gate once under item 4 (both commands plus the 3c-vs-row-C
  fallback) and reduce item 2's rule to "clears item 4's gate (below), unchanged" plus the one
  genuinely item-2-specific clause. Make item 4's σ=0 row read `| 0 | row C | row C | — |`.
  Derive the flag from the prose explicitly: "within 3 points of row C's 88.2%, i.e.
  `--check 0.852`". Correct line 330 the same way.

### M17 — The burst-sequence caveat is duplicated, and the copy in "Pinned numbers" is attached to rows it cannot affect
* **Scope:** lines 51-53 vs 204-206
* Verified on disk: `Burst01`…`Burst16` exist only under `data/processed/real/` (16
  `phone_real_IMG_20191221_104738_Burst*` files); `data/holdout_newdb/**` contains no `Burst*`
  files at all. Rows A-D are drawn from `data/holdout_newdb` and `data/holdout_hard_mined`, so
  "the true intervals are wider than shown wherever those files are included" (52-53), sitting
  directly under the per-row confidence intervals at 46-49, implies a defect in rows A-D that
  does not exist. The only row it can touch is E. The same caveat is stated correctly and
  usefully at 204-206, where it governs item 2c's number.
* **Recommendation:** delete 51-53; if the row-E point is worth keeping, add a clause to the
  row-E paragraph at 55-56. Keep the item 2c copy as the single home.

### M18 — The `--check-recall` rationale is stated three times, and the duplicate carries two numbers the σ table contradicts
* **Reviewer Lens:** `egm-structural` (S6) + `egm-style` (S5)
* **Scope:** lines 143-145, 322-325, 498-499
* Three statements of one rationale. The item 4 copy adds "real-specificity *rises* to 98%"
  (323) — a figure appearing nowhere else and disagreeing with item 4's own σ table: at σ=16,
  accuracy 75.3% and AI-recall 55.7% on a balanced 300+300 set (285-288) imply real-spec
  ≈ 94.9%, not 98%. The same paragraph reports "AI-recall 55.3%" (325) where lines 81, 156,
  288 and 322 all say 55.7%, with no statement that 325 is the full-set run and the others the
  300-per-class run. The style lens adds that line 159 calls the reproduction "exact" when it
  agrees to one decimal on a figure carrying ±5.6 — that is "agrees within its interval" — and
  that item 4's gate ("above 70%, up from 55.7%") is therefore stated against a baseline that
  is not the number verified three lines below it.
* **Recommendation:** state the rationale once in item 1 (143-145); item 4 point 1 cites it in
  one clause. Drop the 98% or derive it from the σ table. Pick one recall figure and label the
  other: "up from the full-set 55.3%; the 300-per-class subsample gives 55.7%". Rewrite 159 as
  "reproduces the scratchpad figure to within its interval (55.7% vs 55.7%, ±5.6)".

### M19 — Retracted revision history is embedded in nine executable items, and one embedded reference dangles
* **Scope:** lines 165, 200, 272, 293, 303, 332, 339, 383, 450, 469, 473 vs the changelog at
  484-558
* Eleven inline "revision 1 proposed X and was wrong" passages sit inside the items a reader is
  executing. Two consequences: the reader must parse retracted history to reach the instruction
  (293-304 spend twelve lines on revision 1's unit error before the actual change at 306), and
  the items now carry the same job as "What changed", i.e. two places to update. One reference
  is outright dangling: "the unguarded ±0.24 swing cited in revision 1" (473) — revision 1 has
  no changelog entry (the changelog starts at Revision 2, line 535), so ±0.24 is defined
  nowhere in the document. `comprehension_report.md` finding 12 flagged this; **UNRESOLVED.**
* **Recommendation:** keep the corrected instruction in the item, move the "revision N was
  wrong" narrative to "What changed". Where the history is load-bearing for a decision (line
  383's "the 30 → 45 experiment was run and never recorded" genuinely is), keep it and drop the
  revision label. Define or delete ±0.24.

### M20 — Every `train.py` citation at or after line 52 is off by +4, caused by item 0's own commit
* **Reviewer Lens:** `egm-structural` (S12) + `egm-design` (N-4) — same finding, independently
  derived
* **Scope:** lines 110, 252-253, 270, 381, 426; plus line 474
* The four-line comment inserted at `train.py:52-55` by item 0's fix shifted everything below
  it. Verified in the current tree: `RandomContrast` is at `:56`, not `:52`;
  `unfreeze_for_fine_tuning(model, trainable_layers=45)` is at `:196`, not `:192`; the
  `find_best_threshold` call is at `:224` and the `THRESHOLD_PATH` write at `:225`, not
  `:219`/`:221`; `evaluate_and_save_metrics` is called at `:228`, not `:224`; the nested
  metrics dict is written at `:152` (built around `:139`), not `:135`. Citations before line 52
  (`:32-45`, `:47`) are still exact. Separately, and still uncorrected from prior N3: the
  `best < default + 0.02` guard is at `calibrate_thresholds.py:75-78`, not `:80` (which is the
  clip) — cited at line 474. All non-`train.py` pointers checked out: `data.py:26`,
  `data.py:29`, `inference.py:13`, `index.html:369`, `.gitignore:6`, `.gitignore:9`,
  `main.py:23`. The document claims these were code-verified (line 537), so the systematic
  drift is worth one pass.
* **Recommendation:** re-resolve the eight `train.py` references and the
  `calibrate_thresholds.py` one. For the citations inside the block item 4 is about to edit
  again, cite symbols rather than line numbers.

### M21 — Item 4's "while editing that block" instruction contradicts item 0 and mislabels `RandomZoom`
* **Reviewer Lens:** `egm-structural` (S13) + `egm-style` (S3)
* **Scope:** lines 311-312 vs line 121 and `train.py:56`
* > "While editing that block, check `RandomContrast(0.1)` and `RandomZoom` — they operate on
  > the same pre-normalised tensors and carry the identical unit hazard."

  Three defects in one sentence. (a) `RandomContrast` was already fixed by item 0 190 lines
  earlier (`train.py:56` now reads `RandomContrast(0.1, value_range=(-1, 1))`), so the
  instruction is dead. (b) The sentence asserts `RandomZoom` "carr[ies] the identical unit
  hazard" while line 121 of the same document says "`RandomZoom` and `RandomRotation` do not
  have this fault" — and rev2 NB1 established the latter. (c) "pre-normalised" is ambiguous and
  reads most naturally as "before normalisation", the opposite of what the document means,
  which is precisely the confusion the unit section exists to prevent.
* **Recommendation:** delete 311-312, or replace with: "`RandomContrast` carried this hazard
  and item 0 fixed it. `RandomFlip`, `RandomRotation` and `RandomZoom` have no absolute-range
  argument, so nothing further to check — but any *new* layer added here must be given the
  [-1, 1] range explicitly."

### M22 — No Problem Statement and no Goals: the plan never states its own success condition
* **Scope:** lines 1-8
* Mapping against the feature-spec standard: **Non-Goals — strong and locatable** (two
  dedicated sections: "Measured non-problems", 83; "Out of scope", 456). **Alternatives
  Considered — present in substance, not locatable** (item 2 vs item 4, 6a/6b/6c, the three
  out-of-scope rejections, item 7's "pick one"; all with stated trade-offs, none attributed to
  an unlinked maintainer, so the standard's Alternatives Rule **passes on substance**).
  **Risks & Mitigations — present but scattered,** and see M6/M10/B3/M7: the document covers
  the current system's failure modes in unusual depth and its *own proposals'* failure modes
  only sporadically, so this area **fails** on the proposed-change half. **Problem Statement
  and Goals — absent entirely.** The front matter is a revision note (3-5) and a status line
  (7-8). Nothing states what the plan is trying to achieve or how a reader knows it is
  finished; the closest surrogates are the "Known bypass" table row (81) and item 4's evidence,
  both of which a reader meets only after 80 lines.
* **Recommendation:** three lines after line 8, e.g. "Goal: close the σ=16 grain bypass
  (AI-recall 55.7% → >70%) without losing more than 3 points of clean accuracy, and keep one
  reproducible producer for every published number. Done when items 2-4 have resolved and the
  table, README and REPORT agree." The structural lens **explicitly declines** three additions
  the standard would nominally imply: a "Proposed Architecture" section (the plan introduces no
  new components), a consolidated "Alternatives Considered" section (it would duplicate items
  2/6/7 and Out of scope), and a consolidated "Risks" section with restated text — if risks
  should be discoverable, add a four-line **pointer** list, no copied prose, or M16-M18's drift
  problem reappears.

## Clarity and confidence calibration — `egm-style`

### M23 — Item 0 is DONE but still gives orders in the modal future
* **Scope:** lines 115-121
* > "Fixed by passing `value_range=(-1, 1)`." … "**It must land before the item 3c
  > re-baseline**, or the baseline measures a model trained on mutilated data."

  This is the same defect class revision 4 claims to have fixed in item 1, in the item directly
  above it. The fix has landed. "It must land before X" asks the reader to schedule work that is
  already done, and buries the instruction that *is* still live: the weights were trained before
  the fix, so 3c must retrain first. (This is the wording surface of M1.)
* **Recommendation:** "This landed before any re-baseline was attempted. The current weights
  were still trained with the fault, so item 3c's baseline must come from a run made after it."

### M24 — "File format is not a cue" is a bolded negative existence claim on an interval that admits a cue
* **Scope:** lines 92-100
* The evidence is honestly reported and then over-read. A CI of [-0.049, +0.060] on a mean-score
  gap is an absence of *measurable* effect, not the absence of a cue; the heading converts it
  into a negative existence claim, in bold, under a section heading that reads "do not spend
  effort here". The corroboration is also weaker than stated: "barely moves" is a 0.3-point
  whole-set accuracy change (88.3 → 88.0), a different quantity from a per-subset mean score —
  and per M11 it is diluted ~7:1.
* **Recommendation:** retitle "**No measurable format cue, despite a skewed corpus**", close
  the bullet with the honest bound ("the interval admits a cue of up to ~0.06 in mean score;
  nothing larger survives"), and replace "barely moves" with the number.

### M25 — Item 2a is written in the indicative, so an unrun experiment reads as a result
* **Scope:** lines 186-191
* > "The root cause of the bypass is that the *relative* noise floor has become a cue …
  > Injecting a fixed σ≈8 grain into *every* input before scoring destroys that differential
  > symmetrically. The attack becomes the preprocessing. Costs some clean accuracy…"

  What has been measured is a bypass (σ=16 → recall 55.7%) and a mean-shift asymmetry (−0.288
  vs −0.081). That supports a mechanism; "the root cause … is" asserts it as established. The
  proposed remedy is then in the present indicative — "destroys that differential
  symmetrically", plus the aphorism "The attack becomes the preprocessing" — which is the mood
  of a finding, not of an experiment the plan has not run. A reader skimming item 2 comes away
  believing 2a works. The next clause flips to asserting an unmeasured cost with a hedge word
  doing a number's job.
* **Recommendation:** "The measured asymmetry is consistent with the *relative* noise floor
  having become a cue … 2a tests whether injecting a fixed σ=8 into every input removes the
  differential: if it does, σ=16 recall should recover with no retrain. Expect a clean-accuracy
  cost and a shifted score distribution, so re-measure clean accuracy and re-fit the
  threshold." Delete "The attack becomes the preprocessing".

### M26 — "Seed it per request" does not fix the nondeterminism it is offered to mitigate
* **Reviewer Lens:** `egm-style` (S8) + `egm-design` (N-2)
* **Scope:** lines 226-228
* > "injected noise makes serving non-deterministic, so the same upload can return different
  > scores. Seed it per request or accept the variance deliberately."

  Seeding *per request* produces a fresh seed per request, which is exactly the
  nondeterminism described. The fix rev2 NM9 named is seeding from the image bytes. As written
  the reader implements a no-op and believes the hazard is closed — the clearest single case in
  the document of wording that causes wrong execution.
* **Recommendation:** "Derive the noise seed from a hash of the image bytes, so the same upload
  always scores the same, or accept the variance and say so in the UI."

---

# 🟢 Nits

## Confidence calibration — `egm-style` (cluster N1)
Each needs calibrating, not deleting.

* **Line 22** — "HTTP rows additionally **prove** the serving path is wired correctly" / "Both
  use identical scoring". Rows A and C are the same dataset at the same threshold and disagree
  by 4.0 points of AI-recall (84.0 vs 88.0); the glossary asserts an interchangeability the
  table contradicts. → "exercise the serving path end to end"; add "rows A and C are different
  subsamples and differ by 4 points of recall; neither is a re-measurement of the other."
* **Lines 113-116** — "**54.5% of each augmented image was blanked to black**" states a sample
  mean as a per-image constant (the code comment at `train.py:55` says "roughly 55%"); and
  "Verified: the output range is restored…" names no check anyone can re-run. → "Measured on one
  augmented batch: 54.5% of pixels, i.e. everything below mid-grey, was clamped to 0.0", and
  name the verification.
* **Line 281 + table 283-288** — "Monotonic and **significant** from σ=4"; column "−3.0,
  significant". No test named, no paired interval given; the only intervals in the document are
  marginal (±2.6), against which −3.0 is borderline. **rev2 NB5 is half-resolved**: revision 4
  fixed the *word* ("real" → "significant", line 516) and left the basis missing. → one clause:
  "deltas are paired on the same 600 images; the interval on a paired difference is tighter than
  either marginal ±2.6" — or drop "significant" and print per-σ counts. **Line 49**'s "Any
  acceptance gate narrower than the relevant interval is undecidable" is stated as a general law
  then applied to differences, where it does not hold; the same sentence fixes both.
* **Lines 383-385** — "**The 30 → 45 experiment has already been run and its result was never
  recorded**". The evidence is a code value and an unused default; that someone *ran a
  comparison* is an inference. → "The code was changed from the 30 default to 45 with no
  recorded comparison, so what 45 buys is unknown."
* **Lines 405-406** — "drop-in change to `build_model`". **rev2 NM10 UNRESOLVED**: this is false
  while preprocessing lives in `data.py:26`, and "drop-in" makes the reader under-budget the
  change. → "backbone swap plus a preprocessing migration (`data.py:26`,
  `load_image_for_inference`, `predict_paths`). Unmeasured."
* **Lines 471-473** — "its `best < default + 0.02` guard **demonstrably** fired". Strong
  circumstantial evidence, not a demonstration. → "the equal accuracies in `thresholds.json:5-6`
  only occur on the guard branch (`calibrate_thresholds.py:75-78`), so the guard fired."
* **Lines 448-451** — "**untracked, and absent from every commit in history** — verified" is the
  author's verification in the same bolded register as code-checked facts, and "no rotation is
  needed on repository grounds" still reads as "no rotation needed", exactly as **rev2 N13**
  predicted — **UNRESOLVED.** → add one clause: "rotate if the key is still live — that is an
  account question, not a repository one."
* **Line 309** — "Blur is optional and carries little of the value." Nothing measured blur. →
  "Blur is unmeasured; skip it."
* **Line 403** — "hand-crafted noise statistics are easy to attack directly": assertion, no
  basis. Keep only with the actual argument — "an adversary can match real-photo noise
  statistics directly" (and see M10).

## The document breaks its own pinning rule twice — `egm-style` (N2)
Line 32 sets the rule ("Every accuracy figure in this document refers to a row below"). Then
**line 17** "so ~75% there is expected, not a regression" (row B is 75.5% ±6.0, and "expected"
is asserted with no basis) and **line 350** "This is where 88% is most likely to collapse"
(row A's 88.0%? row C's 88.2%?). Two unpinned headline figures, one of them in the glossary
revision 4 added to fix this exact class of problem. → "row B's 75.5%" and "row C's 88.2%".

## Terminology and unit drift — `egm-style` (N3) + `egm-structural` (S20)
* **grain / noise / σ / sensor grain / added grain** — the plan's central object has four names
  and no glossary entry. 2a is "Grain-floor normalisation", 2b is "Noise-level test-time
  augmentation", item 4 is "Close the grain bypass" but instructs "Add Gaussian noise". A reader
  can reasonably implement 2a and 2b with different operations. → add a glossary row: "grain =
  i.i.d. Gaussian noise at stated σ; σ is in 0-255 pixel units unless marked 'preprocessed'",
  then use "noise" throughout.
* **real-spec / real-specificity / real-class specificity / specificity** — four forms (20,
  40-44, 68, 144, 323, 342); the script prints `real-specificity`. → one term, including the
  table header.
* **AI-recall / recall** — 19, 68, 219, 324. → "AI-recall" everywhere; bare "recall" next to a
  class named Real is avoidable ambiguity.
* **band / `low_confidence` / "0.45-0.65 band" / `uncertain_low`-`uncertain_high`** — three names
  in the document plus the artifact's own two keys (`thresholds.json:3-4`). → name the artifact
  keys once in the glossary row.
* **"free"** — "Free experiments" (179), "for free" (343), "needs no training at all" (390),
  against "cost an afternoon" (183) and "Costs some clean accuracy" (190). → define once: "free
  = no retrain".
* **σ injection point** — item 1 applies perturbations "**before** preprocessing and before the
  224 resize" (139); item 4's table maps pixel σ to Keras stddev for a post-resize augmenter
  (297-301). One symbol, two injection points, no marker. This is the wording surface of M4.
* Em-dash usage is inconsistent (true em-dashes mostly, bare hyphens at 21 and 261). Row D's
  "Images" cell is "1600" where A-C carry "(150+150)" splits.

## Imperatives that do not resolve to an action — `egm-style` (N4)
* **Lines 265-266** — after an "Either X or Y", "**Do not leave both**" reads as "both fixes",
  the opposite of the intent. → "Pick one. The unacceptable outcome is leaving the artifact
  untracked *and* the constant in place."
* **Line 191 / 224** — "so re-measure both" is glossed 33 lines later. Say it where it is said;
  delete the gloss.
* **Lines 441-442** — "Consider deleting the field — … decide it deliberately rather than in
  passing" instructs the reader to decide with no criterion, so the reader does nothing (rev2
  flagged the same deferral). → "Delete `confidence` from `inference.py`, `main.py`,
  `predict.py` and the UI; keep `ai_probability` as the only headline number. Four files."
* **Lines 240-242** — three options, no selection criterion: the wording half of **B3**.
* **Line 219** — "rather than treating it as a pass": "it" = the experiment, two clauses back.
* **Line 36** — "written by `train.py`, which is itself the defect item 7 records": "which" has
  three candidate referents. "**Row E is neither**" answers a compound property stated as one
  clause.
* **Line 432** — "a badly-fitted one" where the value is known. → "replacing 0.55 with 0.50".

## Precision — `egm-style` (N5)
Honest estimates, keep: "100–200 AI images" (367), "cost an afternoon" (183). Evasions, replace:
* **392** "a handful of scalar high-frequency statistics" — three are then listed → "three".
* **399** "for a few dozen lines" and **370** "a ~20-line synthetic save-path transform" — the
  second is specified operation-by-operation two lines later. Drop both line counts.
* **373** "Keep **perhaps** 50" — the author is choosing this → "Keep 50."
* **242** "`.cache()` … recovers **most of it**" → state the mechanism: "caching the decoded
  tensors removes the per-epoch decode after the first epoch."
* **87** "320 neutral at 2× cost" has no number for "neutral", while "448 **−4.3 pts**" is bolded
  with no interval. 85-91 gives six measurements with two numbers and zero intervals, under a
  heading that forbids further work. → one interval per bullet, or one sentence: "all deltas
  below are within row C's ±2.6 unless a figure is given."
* **240** "≈ **29 GB** before batching" — **rev2 N4**: the true peak is ~2× (list, then
  `np.array` copy, then `train_test_split` copy), and "before batching" does not cover it. →
  "≈29 GB resident, roughly double at peak during construction".
* **199** "non-1024" is still undefined after the glossary was added (comprehension finding 12).
  → "not 1024×1024, unlike the rest of the corpus".

## Hedging where a decision or number exists — `egm-style` (N6)
* **183-184** "may make it unnecessary" → "If either clears the bar at line 216, item 4 becomes a
  preprocessing change."
* **188** "'clean' **partly** reads as 'AI'" — the mean shifts are measured; use them.
* **344** "**largely** re-testing answered questions" → "It re-tested four axes the non-problems
  section settles."
* **353** "the model **may** have **partly** learned 'non-square → real'" followed immediately by
  a bolded unmeasured prediction and "which is exactly the bypass someone would stumble into by
  accident" — two confidence levels in one sentence pair. → "If it has, the predicted failure is
  specific and testable: AI images cropped to phone aspect get called Real. Nothing has measured
  it."
* **400-401** "**very** unlikely to pay for itself" — fine as a forecast; drop "very".
* **291** "nearly four times harder" for a ratio of 3.56 (**rev2 N7 UNRESOLVED**) → "3.6×", and
  name the σ and dataset the shifts were measured at.

## Bold saturation, aphorisms, meta-commentary — `egm-style` (N7)
113 of 558 lines contain bold, carrying at least five different jobs: paragraph labels
(**Change.**, **Evidence.**, **Acceptance.**), warnings, plain emphasis, and arbitrary single
numbers — "**±6.0**" bolded among four intervals that are not (47), "**29 GB**", "**48,707
images**", "**last 45 layers**", "**not**", "**neither**". At this density it marks nothing.
Cheapest fix that keeps the structure: keep bold for the paragraph labels and the three or four
genuine hazards (3b's "The served threshold is a value no code in the repo can regenerate", the
pre-fix warning at 58), strip it from every bare number and negation.

Aphorisms: "a gate that cannot be failed honestly is not a gate" (333) earns its place — it is
the justification for the 3-point choice and compresses it well. "The attack becomes the
preprocessing" (190) does not; see M25.

Meta-commentary addressed to past reviewers rather than to an executor: line 8 "There is no
separate priority list."; 12 "Terms used throughout, defined once here."; 33 "Nothing else is a
headline number." (restates 32); 458 "Listed so they are explicitly declined rather than quietly
dropped."; 161-166 "**Why this item existed.**" — six lines of archaeology inside a DONE item
that the changelog already records. Deleting all six passages costs the reader nothing and removes
~15 lines.

## Compression so tight the meaning is lost — `egm-style` (N8)
* **62-64** "the current model and these rows continue to describe each other correctly" — "describe
  each other" is not a relation a reader can act on, and "the current model" is ambiguous after
  item 0 (code fixed, weights not). → "If item 2 succeeds and no retrain happens, the served
  weights are still the ones these rows measured, so the rows stay valid."
* **148-149** → "It catches regressions across a whole set. It is not the two-image sanity assert
  item 7 also asks for."
* **261-263** "identical to the local file - **by coincidence**, because the constant was set to the
  same value as the artifact" is self-contradictory: a deliberate duplication is not a coincidence.
  → "identical only because the constant duplicates the artifact's value; nothing keeps them in
  step."
* **274-275** "**The retrain / runs twice**" — sentence broken mid-clause by the line wrap, inside
  bold.

## Small leftovers — `egm-style` (N9) and `egm-structural` (S16-S21)
* **"all four" should be "all five"** (104, 461-462). The non-problems list has had five bullets
  since revision 3 added the PNG/JPEG one — and that bullet is the one a spectral branch most
  directly invalidates.
* **Line 447** "Item 1's script can carry this as an `assert`" contradicts 148-149 of the same
  document, and **rev2 N10** noted that an assert inside a multi-minute evaluation requiring
  gitignored `data/` is not a smoke test. Delete the clause.
* **Lines 55-56 vs the table** — row E is pinned as a headline row and simultaneously labelled
  "Treat it as historical". If it cannot be quoted, say so in the table's Dataset cell.
* **Row E's support is unbalanced and the table hides it** (`egm-design` N-5). `metrics.json:20, 26`
  give 520 real / 480 fake. Revision 4 correctly added "single forward pass at 0.50" (prior NM3)
  but not the split, which is the other half of why row E is not comparable to balanced rows A-D.
* **Glossary is `###` directly under the `#` title** (10) while every other top-level section is
  `##`. Skipped level; in a generated TOC it renders at depth 3, below `### Measured non-problems`
  which is correctly a subsection. → `## Glossary`.
* **The Glossary holds content belonging to three other sections** (`egm-structural` S9). Lines
  25-27 (FastAPI app, `static/index.html`, `predict.py`, Docker Space) are project context, not
  term definitions, and sit after the table with no heading — while "Where the project actually
  stands" (73-81) still never mentions that the thing is deployed. Revision 4's changelog (519-520)
  records this gap being found; the fix landed in the wrong section. Line 17's "~75% is expected,
  not a regression" is gate material, not a definition. Line 23's `REPORT.md` entry is the only
  place the document says *what* is wrong with `REPORT.md`, while the item that fixes it (174) just
  says "reconcile". → move 25-27 into "Where the project actually stands"; move the ~75% judgement
  beside row B; move the `REPORT.md` contradiction into the reconciliation item (M9).
* **"(see the end)"** at 104 points at "Out of scope" (456), but the literal end is the 75-line
  changelog, and line 5 already uses "at the end" to mean the changelog. Two targets, one name. →
  name the section.
* **Intra-document pointers are bare item numbers with no links** — the natural companion to M15.
* **Item 1's usage block (131-134) is incomplete**: `eval_holdout.py` also accepts `--batch-size`
  and `--progress` (`:145-146`). List them or label the block "main flags".
* **Changelog size and granularity** (484-558): 75 lines, 13% of an executable plan, three
  revisions deep, and several revision-4 entries restate what the items now say (498-499 duplicates
  143-145; 513-516 duplicates 35-36 and 55-56). Placement at the end is right and per-revision
  structure is defensible for one revision. → keep Revision 4 inline, compress Revisions 2 and 3 to
  one line each pointing at `docs/review/`, which holds the detail. The `docs/review/*` pointers at
  3-5 all resolve.
* **The Platt/isotonic rejection is right about labels and wrong about the band** (`egm-design`
  N-1, lines 464-468). Monotone calibration cannot change a verdict at the correspondingly mapped
  threshold — correct. But the served output also includes a **fixed-width** band
  (`inference.py:44-45`, 0.45-0.65) and `low_confidence` is user-visible (`main.py:69`,
  `index.html:419`). A monotone remap changes which images land inside a fixed-width band, so it
  does change output. The rejection survives; narrow the claim to "cannot change a single Real/AI
  label".
* **2b's serving cost on the batch path is understated** (`egm-design` N-3, line 196). The 4-vs-8
  pass comparison against rotation TTA is fair. Not mentioned: `main.py:91-94` loops uploads
  **serially** on a free CPU Space, so doubling per-image passes doubles batch latency linearly.
  One clause, since batch scan is an advertised feature (`README.md:9, 16`).
* **Item 4's cheap-path fallback describes a branch in which item 4 does not run** (`egm-design`
  N-6, line 330). Line 218 says that on the cheap path item 4 *becomes* the preprocessing change,
  whose gate is item 2's rule — which already uses 0.85. → "on the cheap path, item 2's decision
  rule *is* item 4's acceptance; this section applies only to the retrain path."

---

# Prior findings claimed or implied resolved that are NOT

Listed because per the Stage 7 brief these outrank new findings.

| Prior finding | Source | Status | Here |
| --- | --- | --- | --- |
| NB3 — 3b gated behind the wrong branch | rev2 (Blocker) | **Unresolved, unacknowledged in any changelog.** rev2's own change list called the fix "one cut-and-paste". | B2 |
| NB4 b3 — 2a/2b not expressible with one `--noise` flag | rev2 | **Unresolved.** Revision 4 added the numeric bar but not the instrument. | B1 |
| NM7 — 3a's options not interchangeable | rev2 | **Unresolved.** Highest wasted-compute risk in the plan. | B3 |
| NB4 b2 — pre- vs post-resize σ | rev2 | Documented in code comments, never reconciled with item 4's training range. | M4 |
| NB4 b4 — `--filter` for item 2c | rev2 | **Not implemented**; item 2c is unexecutable. | M11 |
| NM6 — `GaussianNoise` cannot do per-example σ in a `Sequential` | rev2 | **Unresolved across two revisions.** | M5 |
| NM8 — threshold fix scheduled after the retrains it affects | rev2 | **Unresolved** (conflict now named at 433; ordering constraint still unstated). | M7 |
| N1 — "Out of scope" both declines and prescribes work | rev2 | **Unresolved.** | M7 |
| NM2 — table pins values, not procedures | rev2 | **Unresolved**; "sole producer" false for 4 of 5 rows; A/B/D orphaned. | M8 |
| NM4 — README attribution misdiagnosed | rev2 | **Unresolved** (the `REPORT.md`-in-scope half *is* resolved). | M9 |
| Comprehension #1 — item 1 DONE with open work | Stage 6 | **Half resolved**: tense fixed, structure not; revision 4 added line 7, which inherits it. | M14 |
| Comprehension #12 — ±0.24 cited from a document the reader lacks | Stage 6 | **Unresolved.** | M19 |
| N3 — `calibrate_thresholds.py:80` cited for the `+0.02` guard | rev2 | **Unresolved** (it is `:75-78`). | M20 |
| NB5 — paired-comparison convention for "significant" | rev2 | **Half resolved**: the word was fixed, the basis is still missing. | Nit N1 |
| NM10 — "drop-in change to `build_model`" | rev2 | **Unresolved.** | Nit N1 |
| N13 — `gemini_key.txt` reads as "no rotation needed" | rev2 | **Unresolved.** | Nit N1 |
| N4 — 29 GB is the resident figure, not the peak | rev2 | **Unresolved.** | Nit N5 |
| N7 — "nearly four times" for 3.56 | rev2 | **Unresolved.** | Nit N6 |
| NM3 — row E's method label | rev2 | Resolved for method (single pass at 0.50); the 520/480 split is still hidden. | Nit N9 |
| NM9 — item 2's missing numeric bar | rev2 | **Resolved**; the instrument half became B1. | B1 |
| N10 — assert inside the eval script is not a smoke test | rev2 | **Unresolved** (line 447). | Nit N9 |
| M4/rev1 — hold the 710 out during the retrain | rev1, rev2 | **Dropped twice without comment.** | M13 |
| N14 — item 5's collection scheduled too late | rev2 | **Unresolved.** | M12 |

**Verified genuinely resolved** (so not re-raised): B2/unit table, M3/layer count 45, M5/missing
data cell, M6/eager loader, M9/Platt and `confidence`, M10/frequency branch declined, M13/cheap
spectral probe, N1/`gemini_key.txt` commit history, N2/altitude, N4/numbering, NB1/`RandomContrast`
(fixed in code and verified), NB5's confidence intervals and the 3-point tolerance, NM1's item-7
contradiction (now stated as alternatives), NM5's 2,599 total, and Stage 6 findings 1-2, 4-9, 11-13,
16-17. Structurally verified clean: heading hierarchy below the Glossary; every glossary term is
actually used; all three `docs/review/*` pointers resolve; every item-to-item pointer has an existing
target; all non-`train.py` file/line references check out; all markdown tables and fenced blocks are
well-formed.
