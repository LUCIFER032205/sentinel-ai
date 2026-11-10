# Goldfish Review Report — Stage 5 (Initial Zoom Out)

**Target:** `E:\ML mini-project\docs\architecture-improvement-plan.md` (lines 1-174)

**Standard applied (`egm-zoom-out`):** `resources/document_standards/zoom-out-standard.md`

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-zoom-out` | completed |
| `egm-inventive-catalyst` | completed (persona resolved from `agents/inventive_catalyst.md`; the `agents/egm-inventive-catalyst.md` path does not exist in the installed plugin) |

Both reviewers explored the repository and verified document claims against code. No required file was unreadable.

## Severity Breakdown

| Severity | Count |
| --- | --- |
| 🔴 Blocker | 2 |
| 🟡 Major | 13 |
| 🟢 Nit | 4 |

## Lens Summaries

**`egm-zoom-out` (simplicity and scope).** The document is not over-scoped for a student mini-project — it is *mis-weighted*. "Measured non-problems" (lines 22-30) is the strongest section in the document, because it is a list of work not to do. But the cheapest highest-value item is absent entirely (get the measurement harness into the repo), items 3 and 5 propose machinery disproportionate to a payoff the repo's own artifacts show is near zero, and item 4's "cheapest first" ordering rests on a wrong fact.

**`egm-inventive-catalyst` (contradiction elimination).** The plan accepts four trade-offs it could dissolve rather than pay: that robustness requires a retrain, that threshold stability requires more code, that real-world evaluation requires data collection, and that spectral signal requires a second network branch plus a resolution increase. It also carries two unnamed assumptions — that noise-invariance training and a spectral branch can share one backbone, and that `load_dataset`'s fully-resident array imposes no ceiling on data or resolution.

The two lenses converge on the same top conclusion from opposite directions: the single evaluation script that is missing from the repo is both the cheapest item available (zoom-out) and the item that makes most of items 2 and 3 evaporate (inventive catalyst).

---

## Prioritized Findings (Blockers → Major → Nits)

### B1. Measurement harness lives outside the repo; no acceptance criterion in the plan is executable — Severity: 🔴 Blocker
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (independently raised; zoom-out at Blocker, catalyst at Major)
* **Affected Scope:** lines 3-8, 19-30, 40-45, 67-69, 83-84, 108-109, 154-157
* **Discrepancy Description:** The plan's entire authority is "ordered by measured evidence", and the acceptance test for its top item is "Re-run `scratchpad/grain.py`". No `scratchpad/` directory exists in the repo, and nothing in the tree can score the holdout sets — the only `evaluate` call sites are `train.py:114`, `train.py:121` and `train.py:224`, all on the internal train/test split. The holdout images do exist (2,257 files under `data/holdout_*`), but no code reads them. Item 6 calls the scratchpad scripts "throwaway"; they are in fact the only evidence-generating code in the project. The author in three weeks cannot execute this plan, let alone a reader.
* **Rationale & Recommendation:** This is also the document's clearest missed reuse. The pieces already exist: `calibrate_thresholds.py:30-52` `predict_paths(model, paths, batch_size)` already does batched flip-averaged scoring over a path list — the exact contract the served detector uses (`inference.py:77-78`) — and `src/ai_image_detector/data.py:16-19` `iter_image_files` already walks a class folder. Add an **item 0**: one `eval_holdout.py <folder> [--noise σ] [--jpeg q] [--scale f]` printing accuracy, AI-recall and real specificity at the live threshold. That single script is item 1's acceptance test, item 2's reporting mechanism, item 3's stability check, the regenerator of the status table and non-problems section, and it subsumes item 6's `test_inference.py` ask as one `assert` at the bottom. It belongs at position 1 of the suggested order, ahead of everything.

### B2. Item 1's noise σ is in the wrong units — the proposed constant is ~4× too weak for the attack it targets — Severity: 🔴 Blocker
* **Reviewer Lens:** `egm-inventive-catalyst` (Blocker), `egm-zoom-out` (Major) — merged at the higher severity
* **Affected Scope:** item 1 code block (lines 54-62), acceptance test (lines 67-69), σ table (lines 40-45)
* **Discrepancy Description:** The σ table is in 8-bit pixel units (σ=4/8/16 of 255). But `data.py:22-26` applies `preprocess_input` at *load* time, so tensors are already in MobileNetV2's [-1, 1] range before `train.py:59-62` runs the augmenter. `GaussianNoise(0.03)` in that space is ≈0.03 × 127.5 ≈ 3.8 grey levels — the σ=4 row, the *weakest* rung measured and already a 3-point drop. The bypass that motivates the whole item is at σ=16 (≈0.125 preprocessed), and the acceptance test only checks σ=8, so the σ=16 case is never verified. The failure path says "halve the σ and retry" without stating which σ.
* **Rationale & Recommendation:** As written, the plan's highest-value fix would train invariance at the least interesting end of its own ladder. State the unit convention explicitly in the document. Sample σ per example across the measured attack range rather than fixing it at the floor — e.g. a `tf.data.map` adding `tf.random.normal(stddev=tf.random.uniform([], 0.0, 0.125))` — and extend the acceptance test to σ=16. Apply it to a fraction of each batch so clean inputs stay in-distribution, since wide-range noise costs more clean accuracy than a fixed small σ. Note in the same pass that `RandomContrast(0.1)` (`train.py:52`) and `RandomZoom` carry the identical unit hazard on pre-normalised tensors.

---

### M1. Item 3's proposed work product already exists in the repo, and its own evidence says delete the subsystem rather than stabilise it — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both, independently)
* **Affected Scope:** item 3 in full (lines 95-109), suggested order step 2 (lines 169-170)
* **Discrepancy Description:** `artifacts/thresholds.json:10` already records, verbatim: *"0.55 = median of a 200x bootstrap over 1000 held-out newdb images (IQR 0.54-0.58). The gain over 0.50 is +0.5pt, CI [-0.1, +1.2]: treat 0.50 and 0.55 as tied, not as an improvement."* So the exact bootstrap item 3 proposes has already been run; the "write the IQR as an honesty signal" deliverable already exists in that file; the recorded IQR is 4 points wide, not the 24 the document cites; and the recorded conclusion is that threshold choice buys nothing measurable. The item also omits two existing dampers: `calibrate_thresholds.py:80` clips to `[0.35, 0.65]`, making the quoted 0.68 unreachable in any written artifact, and the `best < default + 0.02` guard at lines 75-78 demonstrably fired for the current file (`validation_accuracy == validation_accuracy_default_0_5 == 0.893`). The acceptance test at lines 108-109 therefore describes the behaviour of an unguarded arg-max, not of the script as it exists.
* **Rationale & Recommendation:** Writing ~20 lines of resampler to re-derive "0.50 and 0.55 are tied" is the clearest YAGNI violation in the document. Invert item 3 into a deletion: freeze the threshold, cite the existing `tuning_note` as its justification, and delete `find_best_threshold` with `train.py:12` and `train.py:220` — roughly 60 lines and one script removed, strictly better than 20 added, and it removes the search loop as a noise source permanently. Keep the bootstrap as a one-off script run only when the model changes. Drop item 3 from slot 2 and give the slot to item 0.

### M2. The decision threshold is fitted on single-pass scores while serving uses flip-averaged scores — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** not in the document (code defect adjacent to item 3, lines 95-109)
* **Discrepancy Description:** `train.py:220` fits the threshold on `predict_probabilities` output, which is a single forward pass (`train.py:104-111`). Both serving (`inference.py:77-78`) and `calibrate_thresholds.py:46-49` flip-average two passes. The threshold is being fitted on a different score distribution from the one it gates.
* **Rationale & Recommendation:** This is a real defect, it is cheaper to fix than anything item 3 proposes, and the document does not mention it. Add it to item 3's slot in place of the bootstrap machinery.

### M3. Item 4's "cheapest first" candidate rests on a wrong fact — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** status table (line 16), item 4 bullet 1 (lines 121-123), suggested order step 4 (lines 172-173)
* **Discrepancy Description:** Both the table ("frozen base + 30 unfrozen layers") and the candidate ("Currently only the last 30… change one argument in `unfreeze_for_fine_tuning`") are wrong. `train.py:192` calls `unfreeze_for_fine_tuning(model, trainable_layers=45)`. 30 is only the function default (`model.py:45`) and no caller uses it. A reader following the document would edit a default nothing reads and observe no change. The "nearly free to try" experiment has in part already been run — 30 → 45 happened and nothing records the result.
* **Rationale & Recommendation:** Correct both numbers to 45, point at `train.py:192` rather than `unfreeze_for_fine_tuning`, and state what 45 → 60/80 is expected to show relative to whatever 30 → 45 produced. If that earlier result was never recorded, say so — it changes the expected value of the experiment.

### M4. Item 2 proposes collecting data the repo already has — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both, with different follow-through — see M5)
* **Affected Scope:** item 2 (lines 76-91), suggested order step 1 (lines 167-168)
* **Discrepancy Description:** `data/processed/real/` already contains **710 `phone_real_IMG_*.jpg` files** — genuine phone photos, non-square, non-1024, already in the training corpus, and largely the ~11% of non-1024 training data the document mentions at line 79. "No number in this project covers the input users will actually send" (line 80) is true only because nobody has sliced the existing data, not because the data is absent. "Needs data collection rather than code" (line 86) is therefore half wrong, which undercuts the document's own reason for ranking item 2 second.
* **Rationale & Recommendation:** Two cheaper transforms before any collection. (a) *Slice, don't collect*: report accuracy on the `phone_real_*` subset separately — a filename filter, ~5 lines, available today via item 0's script. The number is optimistic because those files are in train, but an optimistic number on the right distribution beats an exact number on the wrong one. (b) *Ride the retrain*: hold the 710 out of training during item 1's retrain, which is happening anyway, and the real-side holdout costs zero collection and zero extra compute. This reverses the dependency argument at lines 86-88 — attaching item 2 to item 1 makes it free. Caveat the sample: the 710 are not 710 independent scenes (`Burst01`…`Burst16` are near-duplicates of one moment), so report distinct-scene count, not file count, and group-split by scene.

### M5. Item 2's genuinely-uncovered axis is misidentified — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst` and `egm-zoom-out` — **two distinct diagnoses, both worth acting on**
* **Affected Scope:** item 2 (lines 78-84) against "Measured non-problems" (lines 22-30)
* **Discrepancy Description (catalyst):** Item 2 frames the gap as *format* — "not square and not 1024px", "downloaded from a chat app, screenshotted, cropped". But the non-problems table 50 lines earlier already measured JPEG q30, 0.25× downscale and screenshot re-encoding (all within noise) and centre-crop/letterbox/squash across "every shape tested" (all tied). If those results hold, the save-path and aspect-ratio halves of item 2 are already answered by the author's own measurements, and the document is proposing to collect data to re-test a dimension it has declared a non-problem. What is genuinely untested is *content and generator provenance*: the fake class is one homogeneous corpus (`data/holdout_newdb/fake/newdb_ai_*`, ~1000 files; `data/processed/fake` similarly uniform by name), and nothing evaluates AI images from a generator the training set never saw. That is where 88% is most likely to collapse.
* **Discrepancy Description (zoom-out):** The 710 phone photos appear to sit in the **real class only**, while the fake class is the square generated corpus. That is a shape/source confound — the model may have learned "non-square → real" — and it predicts a specific asymmetric failure: AI images screenshotted or cropped to phone aspect get called Real. That is sharper and more damaging than "no number covers real uploads", and it names the one data cell that must actually be collected: **non-square AI images**.
* **Rationale & Recommendation:** Both diagnoses point at the fake class, from different angles, and both cut item 2's hand-labour by more than half — which matters when days of collection compete directly with the retrain. Rewrite item 2 as: (a) free — score the existing 710 phone photos for real-photo specificity; (b) collect only the missing cell, ~100-200 AI images that are both non-square/screenshotted *and* from an unseen generator, which satisfies both lenses at once; (c) replace the format half with a ~20-line synthetic save-path pipeline (random non-square crop, phone-resolution resize, chat-app JPEG re-encode) applied to the existing holdout, keeping ~50 genuinely round-tripped images as a fidelity spot-check.

### M6. `load_dataset` is eager and fully in-memory — item 1 may not run, and its baseline is confounded — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both; the document names the constraint nowhere)
* **Affected Scope:** item 1 (lines 34-72), item 2 (line 84), item 4 (lines 132-135), suggested order step 3 (line 171)
* **Discrepancy Description:** `data.py:29-56` decodes and preprocesses every image into one resident float32 `np.array` — ~600 KB per image at 224×224. `data/processed/real` now holds 24,763 files and `data/processed/fake` at least 20,000 (`data/processed/fake/19999.jpg` exists): ~45k × 224×224×3 × 4 bytes ≈ **27 GB of RAM** before batching. Separately, `artifacts/metrics.json` reports a test support of 1,000, which under `train.py:32-45` (30% temp, half test = 15%) implies the last recorded run trained on ~6,700 images. The corpus has grown several-fold since the baseline the plan compares against. `IMAGE_SIZE` is also a module constant (`config.py:15`), not a parameter, so item 4's "revisit the resize" is not the one-line change implied — it moves training memory, model input shape and serving preprocessing together.
* **Rationale & Recommendation:** The plan's two most expansive items are silently gated on an unstated prerequisite. Add a one-line prerequisite to item 1: move `load_dataset` to a `tf.data` file-list with map-decode, or `tf.keras.utils.image_dataset_from_directory` (framework-standard, streams from disk, carries augmentation natively), or pin a fixed subset. Then state explicitly that the retrain must run **twice** — once without noise augmentation on the current data to re-establish the baseline, once with — or the "clean accuracy must not fall below 87%" criterion is meaningless, because the data changed too. Note the real cost honestly: per-epoch decode slows epochs on a disk-bound student machine; `.cache()` to a local file recovers most of it.

### M7. Headline numbers are not pinned, so items 1 and 3 have no acceptance anchor — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, corroborated by `egm-inventive-catalyst`
* **Affected Scope:** lines 3-4, 19, 20, 42 (σ=0 row), 67-69, 79
* **Discrepancy Description:** Five "ordinary accuracy" figures are in circulation, none tied to a stated threshold and dataset: document table "88.0% ordinary" (line 19); σ=0 row "88.2% / 88.0% AI-recall" (line 42); `artifacts/metrics.json` 0.888 at threshold 0.5; `artifacts/thresholds.json` validation_accuracy 0.893; `README.md:19` "~89% accuracy (80% AI recall, 86% real-photo specificity at threshold 0.55)". The AI-recall figures disagree outright — 88.0% in the document vs 80% in the README. Hard-mined is 75.5% (line 19) vs ~73% (`README.md:74`). Sample size is "~3,700 held-out predictions" (line 4) vs `README.md:69`'s 1,600 vs 2,257 files on disk. Item 2's "150/150 real holdout images are 1024×1024" is a 15% sample of the 1,000 files in `data/holdout_newdb/real/`, stated flatly as 100% in `README.md:84`.
* **Rationale & Recommendation:** "Clean accuracy does not fall below 87%" cannot be applied when the pre-change number is somewhere in 88.0-89.3 depending on which artifact you believe. Add one short table at the top fixing dataset, image count, threshold and metric per headline number, make item 0's script the sole producer of those numbers, and reconcile or supersede the README's conflicting figures in the same pass.

### M8. The first retrain will silently overwrite the hand-maintained artifacts the plan quotes — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out` (schema drift independently corroborated by `egm-inventive-catalyst`)
* **Affected Scope:** suggested order (lines 165-174), item 3 (lines 95-109)
* **Discrepancy Description:** `train.py:221` overwrites `artifacts/thresholds.json` and `train.py:224` overwrites `artifacts/metrics.json`. Both files are currently hand-shaped, not machine-generated. `artifacts/metrics.json` is a flat schema with top-level `test_accuracy` / `test_f1_fake` / `threshold`, but `train.py:135-146` writes a nested `{"evaluation", "thresholding", "confusion_matrix", "classification_report"}` schema — the committed metrics were not produced by the current `train.py`. `artifacts/thresholds.json` carries a `tuning_note` field `find_best_threshold` never writes, and its guard demonstrably fired, so the function would have written 0.5, not the 0.55 that is in the file and served (`inference.py:40`). The presence of `artifacts/metrics_before_retrain_20260326_150355.json` suggests this has already bitten once.
* **Rationale & Recommendation:** Make "snapshot `artifacts/thresholds.json` and `artifacts/metrics.json`, and reconcile them with what `train.py` actually emits" a pre-step of the first retrain. The served threshold is currently a value no code in the repo can regenerate — a larger correctness problem than anything item 3 proposes to fix.

### M9. Item 5 is over-scoped for the symptom it names, and leaves three confidence-shaped numbers in play — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** item 5 (lines 139-148)
* **Discrepancy Description:** The stated root problem is "the UI shows it to 2 decimal places, which implies a precision that is not there". Verified and worse: `static/index.html:369` formats as `(value * 100).toFixed(2)` — two decimals *of a percent*, four significant figures of a probability — applied to `ai_probability` (line 418), to `confidence` (line 419), and to both in the batch table (lines 509-510); `predict.py:21-22` does the same with `:.2%`. The cheapest fix for the stated symptom is the format string. Platt/isotonic fitting plus a stored coefficient artifact plus a new inference path solves a different (real but unstated) problem: that the score has no probabilistic meaning. Separately, calibration is *monotonic* — at the correspondingly mapped threshold every label is unchanged, so item 5 cannot move a single verdict, and its placement among improvement items invites the opposite reading. The item also leaves untouched the number most likely to confuse a user: `inference.py:84-88` computes `confidence` as a distance-to-threshold rescaling, which is neither a probability nor a confidence in any statistical sense, and the UI shows it as a headline percentage beside `ai_probability` (`index.html:264, 274`). Two uncalibrated numbers derived from one uncalibrated number is the actual UI problem, and Platt scaling does not fix it.
* **Rationale & Recommendation:** Split item 5. The cheap half is a net deletion: print `ai_probability` to 0 decimals or as a coarse band (Likely real / Uncertain / Likely AI, derived from the existing `low_confidence` logic), and **delete** the derived `confidence` field from `inference.py`, `main.py`, `predict.py` and the UI. Keep the raw value in the JSON response and out of the headline. Defer Platt scaling to the same explicit not-now list as the frequency branch — for a mini-project whose threshold tuning is already within noise, calibrated probabilities are decoration. While in the file, note the duplicated constants: `inference.py:13-14` (`DEFAULT_THRESHOLD = 0.55`, `DEFAULT_BAND = 0.10`) restate the artifact value, and the band is hardcoded again at `calibrate_thresholds.py:83-84`.

### M10. The frequency-domain branch should be an explicit "won't do", not a listed candidate — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** item 4 bullet 3 (lines 126-130) and caveat (lines 132-135)
* **Discrepancy Description:** The document is honest that this is unmeasured and the largest piece of work, but it is still a candidate inside a prioritised plan — which is how a second input branch, a revisited resize, a new preprocessing path and the invalidation of every measured non-problem in lines 22-30 end up being started two weeks before a deadline. The entire measurement base (resolution neutral, squashing fine, flip-averaging wins) is tied to the current single-branch 224×224 spatial model; a spectral branch invalidates all of it at once.
* **Rationale & Recommendation:** Move it out of the candidate list into a short "Out of scope / future work" section with one sentence on why (cost, and it invalidates the existing measurement base). Keep bullets 1-2 as the only backbone candidates.

### M11. Item 1 accepts "needs a retrain" as inevitable; two no-retrain transforms are untested — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** item 1 heading (line 34) and root cause (lines 47-50), suggested order step 3 (line 171)
* **Discrepancy Description:** The document's own root cause is that the *relative* noise floor has become the cue ("clean has partly become the model's cue for AI"). Retraining removes the cue by making the model invariant. Equalising the input's noise floor removes it at inference, for free, and neither option is considered: (a) **grain-floor normalisation** — inject a fixed σ≈8 grain into *every* input before scoring, so both classes arrive equally grainy and the differential cue is destroyed symmetrically; the attack becomes the preprocessing. (b) **noise-level TTA** — score at σ=0 and σ=8 and average; `inference.py:77-78` already stacks two variants into one batch, so this is two extra array ops inside an existing design, not a new serving architecture.
* **Rationale & Recommendation:** Both are testable against the same grain script in an afternoon and would establish whether the retrain — the most expensive item in the plan on a student machine — is needed at all. Run them first and report the result as the floor the retrain must beat. If (a) recovers most of σ=16 recall, item 1 downgrades from "highest value, needs a retrain" to a one-line preprocessing change. Secondary liabilities to state if adopted: (a) costs some clean accuracy and invalidates the current 0.55 threshold, so it needs a re-measure; (b) doubles forward passes from 2 to 4 — still a quarter of the rotation-TTA cost the document already rejected.

### M12. Item 1 and item 4's spectral branch are in direct conflict, and the suggested order makes it worse — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** item 1 against item 4 (lines 126-135); suggested order (lines 171-173)
* **Discrepancy Description:** Noise augmentation explicitly teaches the backbone to **ignore** high-frequency energy. The FFT branch exists to **exploit** high-frequency statistics. The suggested order runs item 1 before item 4, so item 4 would be evaluated on a backbone trained to discard precisely the signal it needs. The document names the clean-vs-robust trade-off (line 71) but never this one; line 135 shows the author was one step from seeing it.
* **Rationale & Recommendation:** Do not force robustness and spectral sensitivity through one set of weights. Either feed the FFT branch the un-augmented tensor while the spatial branch gets the noise, or keep two models and combine logits. State the secondary liability: two augmentation pipelines plus a fusion step, and a two-model ensemble doubles inference cost and artifact size — material for a project shipping a 224px MobileNet specifically to stay cheap. (Interacts with M10: if the spectral branch becomes explicit future work, this conflict is deferred rather than resolved, which is also an acceptable outcome — but it should be named.)

### M13. The cheap spectral probe is skipped in favour of the expensive branch — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** item 4 (lines 126-135)
* **Discrepancy Description:** The document jumps from "unfreeze more layers" straight to a second FFT branch concatenated before the Dense head, and admits none of it is measured. Between those sits a far cheaper probe that item 1's own evidence motivates: a handful of **scalar high-frequency statistics computed on the full-resolution image before the 224 resize** — estimated noise σ, high-frequency band energy ratio, residual variance after a 3×3 median filter — combined with the existing CNN logit by a logistic regression on the validation split.
* **Rationale & Recommendation:** This is structurally better for two reasons. It tests the "spectral artifacts carry signal" hypothesis for a few dozen lines and no backbone change, and it **dissolves the document's own blocker at lines 132-134** ("the 224×224 resize throws away the high-frequency detail a spectral branch would want, so that option likely needs the resize revisited too") because the scalars are computed before the resize — the expensive resolution question never has to be reopened. If the scalars add nothing, the FFT branch is very unlikely to pay for itself. Two liabilities to state: serving must retain the full-res decode, which `load_image_for_inference` (`inference.py:70`, `data.py:59-64`) currently discards inside a single call; and hand-crafted noise statistics are easy to attack, since an adversary can match real-photo noise statistics directly. This buys measurement and a likely accuracy bump, not durable robustness.

---

### N1. The `gemini_key.txt` bullet is already answered, and the key is dead — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** item 6 (lines 158-159)
* **Discrepancy Description:** `.gitignore:6` already lists `gemini_key.txt`, so "confirm it is gitignored" was a five-second check that should have been resolved before the document shipped. Further, nothing in the project references it — a search for `gemini|genai|GEMINI` excluding `.venv` matches only this document and `.gitignore`. It is a dead credential.
* **Rationale & Recommendation:** Reduce to one line: delete the file and rotate the key. Rotation stays mandatory regardless, since deleting the working copy does not remove it from history. Both the gitignore entry and this plan item then become unnecessary.

### N2. Altitude drop in item 1 — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** lines 54-65
* **Discrepancy Description:** A full 8-line Keras `Sequential` block plus a paragraph on 3×3/5×5 kernels applied to "~20% of batches" is implementation detail in a document otherwise correctly pitched at the architectural level. The architecturally meaningful content is one sentence: add Gaussian noise to the augmenter at a σ covering the measured attack range; blur is optional and carries little of the value.
* **Rationale & Recommendation:** Replace the block with that sentence. Note that the code block is also where the unit error in B2 hides — prose would have forced the unit to be stated.

### N3. Broken cross-reference and misleading path description — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-zoom-out`, `egm-inventive-catalyst` (both)
* **Affected Scope:** lines 134-135, lines 154-157
* **Discrepancy Description:** Lines 134-135 cite "item 2's finding (resolution does not currently help)", but the resolution finding is in "Measured non-problems" (line 26); item 2 is the real-world holdout, so a reader chasing the citation loses the thread. Lines 154-157 describe `scratchpad/` as being under the repo when it is outside it.
* **Rationale & Recommendation:** Fix the cross-reference to point at the non-problems section, and state plainly that the scratchpad is outside the repository (which is the subject of B1).

### N4. Item numbering and the suggested order contradict each other — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** lines 6-8, 86-88, 165-175
* **Discrepancy Description:** Lines 6-8 claim the items are "ordered by measured evidence per unit of work". Line 86 then apologises that item 2 is item 2 only for logistical reasons and "is arguably more important than everything below it", and the suggested order puts it first anyway, reshuffling 1/2/3 into 2/3/1. Two competing orderings in one short plan is real friction for a document whose purpose is to answer "what do I do next".
* **Rationale & Recommendation:** Renumber the items to match the suggested order and delete the apologia at lines 86-88; the suggested-order section then collapses into the numbering. B1 adds a new item at position 1 and M4 makes the old item 2 a rider on item 1, so the renumber is happening anyway.

---

## Scope Judgement for the Setting

Both reviewers were asked to judge scope against a student ML mini-project rather than a production team, and both reached the same verdict: **the scope is broadly well-judged; the sequencing is the weak part.** Items 4 and 5 are correctly deferred and correctly labelled unmeasured, item 6's single-assertion test ask is correctly sized, and the "Measured non-problems" section is better practice than most student work. The failures are that the cheapest item is missing entirely, two items propose machinery for work already done or already shown to be worthless, and the retrain — by far the most expensive item on a student machine — is scheduled ahead of the free experiments that would tell you whether it is needed.

## Consolidated Re-ordering Proposed by the Reviewers

0. **New item 0** — `eval_holdout.py`, reusing `calibrate_thresholds.predict_paths` and `data.iter_image_files`. Cheapest item in the plan; nothing else is verifiable without it.
1. **Free experiments against item 1** — grain-floor normalisation and noise-level TTA (M11), and slicing the existing 710 phone photos (M4). Establishes the floor the retrain must beat and the first real-distribution number, at zero collection cost.
2. **Item 1 prerequisites** — snapshot `artifacts/*.json` (M8), fix the eager `load_dataset` (M6), re-baseline on current data.
3. **Item 1** — noise augmentation at a σ stated in preprocessed units and covering σ=16 (B2).
4. **Item 2, reduced** — collect only the missing cell: non-square AI images from an unseen generator (M5).
5. **Item 4 bullets 1-2 only**, plus the cheap pre-resize scalar probe (M13); correct the layer count to 45 first (M3).
6. **Item 6**, plus the cheap half of item 5 (fewer digits, delete `confidence`) and the `train.py:220` threshold-fit fix (M2).
7. **Cut or defer:** item 3 as written (invert to a deletion, M1), item 5's Platt scaling (M9), item 4's frequency branch (explicit future work, M10).
