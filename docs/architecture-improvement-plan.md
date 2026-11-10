# SENTINEL_AI — Architecture Improvement Plan

Revision 3, 2026-11-10. Revisions 1 and 2 were both reviewed and superseded; see
`docs/review/zoom_out_report.md` and `docs/review/zoom_out_report_rev2.md`.
Corrections are listed at the end under "What changed".

**Item 1 is now done** and item 0 (a training bug found during review) is fixed.

Items are numbered in execution order. There is no separate priority list.

---

## Pinned numbers

Every accuracy figure in this document refers to a row below. Nothing else is a
headline number. All rows use flip-averaged scoring at threshold 0.55 unless the
row says otherwise.

| ID | Dataset | Images | Threshold | Accuracy | AI-recall | Real-spec |
|----|---------|--------|-----------|----------|-----------|-----------|
| **A** | `holdout_newdb`, served over HTTP | 300 (150+150) | 0.55 | 88.0% | 84.0% | 92.0% |
| **B** | `holdout_hard_mined`, served over HTTP | 200 (100+100) | 0.55 | 75.5% | 71.0% | 80.0% |
| **C** | `holdout_newdb`, offline, grain test baseline | 600 (300+300) | 0.55 | 88.2% | 88.0% | 88.3% |
| **D** | `holdout_newdb` + `holdout_hard_mined`, offline | 1600 | 0.55 | 83.2% | 80.2% | 86.2% |
| **E** | internal train/test split, `artifacts/metrics.json` | 1000 | 0.50 | 88.8% | 89.0% | 88.7% |

Confidence intervals differ per row and are wider than revision 2 claimed:
row A ±3.7, row B **±6.0**, row C ±2.6, row D ±1.8. Per-class columns use half
the images, so they are wider still — row B's AI-recall is **±8.9**. Any
acceptance gate narrower than the relevant interval is undecidable.

These assume independent images. `data/processed/real` contains burst sequences
(`Burst01`…`Burst16`) that are near-duplicates of one moment, so the true
intervals are wider than shown wherever those files are included.

Row E was produced by an older `train.py` than the one in the tree — its schema
does not match what `train.py` writes today (see item 3). Treat it as historical.

**`README.md` currently quotes row D's recall and specificity beside row A's
accuracy.** Both numbers are real but they come from different sets. Item 1
makes one script the sole producer of this table, and the README must be
regenerated from it in the same pass.

---

## Where the project actually stands

| | |
|---|---|
| Model | MobileNetV2, ImageNet weights, frozen base then **last 45 layers** unfrozen (`train.py:192`) |
| Input | 224×224, aspect squashed, preprocessed to [-1, 1] at load time (`data.py:26`) |
| Scoring | flip-average, 2 forward passes, single threshold from `artifacts/thresholds.json` |
| Training corpus | 24,763 real + 23,944 fake = **48,707 images** in `data/processed/` |
| Known bypass | added grain, σ=16 → AI-recall falls to 55.7% |

### Measured non-problems — do not spend effort here

- JPEG re-compression to q30, downscaling to 0.25×, screenshot-style re-encoding:
  all within noise.
- Inference resolution: 320 neutral at 2× cost, 448 **−4.3 pts** at 4× cost.
- Aspect handling: centre-crop and letterbox both tie with squashing on 4:3, 16:9
  and 9:16. Squashing is fine.
- Rotation test-time augmentation: 8-variant averaging scores below flip-average
  at 4× the cost.
- **File format is not a cue, despite a skewed corpus.** `data/processed/real` is
  0 PNG out of 24,763; `data/processed/fake` is 7,903 PNG out of 23,944, and the
  holdout repeats the skew (0/1,000 vs 267/999). A PNG carries no JPEG artifacts,
  so "no artifacts → fake" is learnable in principle. Measured: PNG fakes score
  0.858 mean p(AI) against 0.848 for JPEG fakes, a difference of **+0.010, CI
  [-0.049, +0.060]** — indistinguishable from noise. The `--jpeg 30` row above
  agrees, since it converts PNG fakes to JPEG and barely moves. Worth fixing as
  dataset hygiene before the next corpus expansion; **not** a reason to distrust
  rows A-D.

These were measured on the current single-branch 224×224 spatial model. A
frequency-domain branch would invalidate all four at once — which is one reason
it is out of scope (see the end).

---

## Item 0 — DONE: training augmentation was destroying images

`RandomContrast(0.1)` at `train.py:52` takes `value_range=(0, 255)` by default and
**clips to that range**. Tensors arriving from `data.py:26` are in [-1, 1], so every
value below mid-grey was clamped to 0.0. Measured: **54.5% of each augmented image
was blanked to black**, and no negative value survived.

Fixed by passing `value_range=(-1, 1)`. Verified: the output range is restored to
[-1, 1], negatives survive, and contrast still varies across draws.

This is not a units subtlety like the one in item 4 — it is destructive clipping,
and it affected every training run that produced the current model. **It must land
before the item 3c re-baseline**, or the baseline measures a model trained on
mutilated data. `RandomZoom` and `RandomRotation` do not have this fault.

---

## Item 1 — DONE: measurement harness is in the repository

`eval_holdout.py` is committed. It is the only code in the tree that reads
`data/holdout_*`.

```
python eval_holdout.py data/holdout_newdb [--noise S] [--jpeg Q] [--scale F]
                       [--limit N] [--seed N] [--threshold T] [--check MIN]
```

It reuses `data.iter_image_files` and `data.preprocess_image`, reads the live
threshold through `inference.load_calibration`, and flip-averages exactly as
`inference.py` does. Perturbations are applied to the decoded 0-255 BGR image
**before** preprocessing and before the 224 resize, matching how the published
numbers were measured; `--check` exits non-zero below a floor, which makes it the
smoke test item 7 asked for.

**Verified against prior measurements** (300 images per class, seed 0):

| Run | Result | Expected |
|---|---|---|
| clean | 88.3% ±2.6 | row A, 88.0% ±3.7 |
| `--noise 16` | AI-recall **55.7%** ±5.6 | 55.7% from the scratchpad run |
| `--jpeg 30` | 88.0% ±2.6 | non-effect |

The σ=16 reproduction is exact, so the acceptance test for item 4 now runs.

**Problem.** Every number above was produced by throwaway scripts in a temporary
directory outside the repository. They are not in version control and will not
survive. No code in the tree reads `data/holdout_*` at all — the only evaluation
runs on the internal train/test split. Revision 1 of this document cited an
acceptance test of "re-run `scratchpad/grain.py`", a path that does not exist
here. **The 2,257 holdout images on disk are currently unreachable by any
committed code.**

Note: `calibrate_thresholds.py:30` `predict_paths` was *not* reused. It calls
`load_image_for_inference`, which decodes, resizes and preprocesses in one call and
leaves no seam to inject a perturbation at the right stage. `eval_holdout.py` has
its own loop for that reason.

**Still to do here:** regenerate the pinned-numbers table and `README.md` from this
script so there is one producer of those figures, and reconcile `REPORT.md`, which
contradicts both.

---

## Item 2 — Free experiments, before committing to any retrain

A retrain on 48,707 images on a CPU-only student machine is the most expensive
thing in this plan. These three experiments cost an afternoon between them and
may make it unnecessary. **Run them before item 4 and report the result as the
floor a retrain has to beat.**

**2a. Grain-floor normalisation (no retrain).** The root cause of the bypass is
that the *relative* noise floor has become a cue — real photos carry sensor
grain, AI images are smooth, so "clean" partly reads as "AI". Injecting a fixed
σ≈8 grain into *every* input before scoring destroys that differential
symmetrically. The attack becomes the preprocessing. Costs some clean accuracy
and would invalidate the 0.55 threshold, so re-measure both.

**2b. Noise-level test-time augmentation (no retrain).** Score at σ=0 and σ=8
and average. `inference.py` already stacks two variants into one batch, so this
is two array operations inside the existing design. Doubles forward passes from
2 to 4 — still half the cost of the rotation TTA already rejected.

**2c. Score the 710 phone photos already in the corpus.** `data/processed/real/`
contains 710 `phone_real_IMG_*.jpg` files: genuine phone photos, non-square,
non-1024. Revision 1 claimed real-world data needed collecting. It does not, for
the real class. The number is optimistic because these files are in training, so
report it as a ceiling, not an estimate.

Caveat: these are not 710 independent scenes — `Burst01`…`Burst16` are
near-duplicates of a single moment. Report distinct scenes, not file count, and
group-split by scene if they are ever held out.

**Decision rule.** If 2a or 2b recovers most of σ=16 recall, item 4 downgrades
from a retrain to a preprocessing change.

---

## Item 3 — Retrain prerequisites

Only needed if item 2 does not dissolve item 4. **These are blocking: skipping
them makes the retrain's result uninterpretable.**

**3a. `load_dataset` cannot load this corpus.** `data.py:29` decodes and
preprocesses every image into one resident float32 array: 48,707 × 224×224×3 × 4
bytes ≈ **29 GB** before batching. Move to `tf.data` with map-decode, or
`tf.keras.utils.image_dataset_from_directory`, or pin a fixed subset. Per-epoch
decode will slow epochs on a disk-bound machine; `.cache()` to a local file
recovers most of it.

**3b. Snapshot the artifacts first.** `train.py:221` overwrites
`artifacts/thresholds.json` and `train.py:224` overwrites
`artifacts/metrics.json`. Both are currently hand-shaped:

- `thresholds.json` holds threshold 0.55 and a `tuning_note` that
  `find_best_threshold` never writes. **The served threshold is a value no code
  in the repo can regenerate.** The first retrain silently reverts it.
- `metrics.json` has a flat schema; `train.py:135` writes a nested one. The
  committed file was not produced by the current `train.py`.
- `artifacts/metrics_before_retrain_20260326_150355.json` suggests this has
  bitten once already.

**3b-bis. The served threshold is not in version control at all.**
`.gitignore:9` ignores `artifacts/*` and un-ignores only `ai_image_detector.keras`,
so `thresholds.json` and `metrics.json` are untracked. The Hugging Face Space builds
from the git push, so it has no `thresholds.json` and falls back to
`inference.py:13` `DEFAULT_THRESHOLD = 0.55` / `DEFAULT_BAND = 0.10`.

That currently produces 0.55 and a 0.45-0.65 band, identical to the local file -
**by coincidence**, because the constant was set to the same value as the artifact.
Retune one without the other and local and deployed behaviour diverge silently with
nothing to catch it. Either track `thresholds.json` (add `!artifacts/thresholds.json`)
or delete the constant and fail loudly when the file is missing. Do not leave both.

**3c. Re-baseline on the current data.** `metrics.json` reports a test support of
1,000, which under the 70/15/15 split implies a training run on roughly 6,700
images. The corpus is now 48,707 — it has grown about sevenfold. Without a clean
re-baseline on today's data, "clean accuracy must not fall below 87%" compares
against a model trained on a different dataset and means nothing. **The retrain
runs twice: once without noise augmentation, once with.**

---

## Item 4 — Close the grain bypass

**Evidence.** Measured on row C's images. Monotonic and significant from σ=4:

| σ (0–255 units) | Accuracy | AI-recall | vs clean |
|---|---|---|---|
| 0 | 88.2% | 88.0% | — |
| 4 | 85.2% | 80.0% | −3.0, real |
| 8 | 82.5% | 72.3% | −5.7, real |
| 16 | 75.3% | **55.7%** | −12.9, real |

Noise moves AI images toward "Real" nearly four times harder than it moves real
photos (mean shift −0.288 vs −0.081).

**Unit convention — this is where revision 1 was wrong.** `data.py:26` applies
`preprocess_input` at load time, so tensors reaching the augmenter are already in
[-1, 1], not [0, 255]. The conversion is **σ_preprocessed = σ_pixel / 127.5**:

| σ in pixel units | stddev to pass Keras |
|---|---|
| 4 | 0.031 |
| 8 | 0.063 |
| 16 | **0.125** |

Revision 1 proposed `GaussianNoise(0.03)`, which is σ≈3.8 pixel units — the
weakest rung on the ladder above, not the σ=16 bypass it was meant to close.

**Change.** Add Gaussian noise to the augmenter in `train.py:47`, sampling σ per
example across the measured attack range (0 to 0.125 preprocessed) rather than
fixing it, and applying it to a fraction of each batch so clean inputs stay in
distribution. Blur is optional and carries little of the value.

While editing that block, check `RandomContrast(0.1)` and `RandomZoom` — they
operate on the same pre-normalised tensors and carry the identical unit hazard.

**Acceptance.** Via item 1's script, on the row C images:
```
python eval_holdout.py data/holdout_newdb --noise 16 --check 0.85
python eval_holdout.py data/holdout_newdb --check 0.86
```

σ=16 AI-recall rises above 70% (from 55.7%), **and** clean accuracy stays within
**3 points** of the item 3c re-baseline. Revision 2 said 1.5 points, which is below
row C's own ±2.6 interval and therefore undecidable — a gate that cannot be failed
honestly is not a gate. Use more images (`--limit` unset) to narrow the interval if
a tighter bar is wanted.

---

## Item 5 — Collect only the missing data cell

**The gap is not format.** Revision 1 proposed collecting real-world images in
phone shapes and chat-app save paths. The non-problems section already shows
JPEG, downscale, screenshot re-encode and every aspect ratio are within noise,
and item 2c covers the real class for free. That proposal was largely re-testing
answered questions.

**The gap is provenance, and it has a confound attached.** Two things, both
pointing at the fake class:

1. Every AI image in the project comes from one homogeneous corpus. Nothing
   evaluates a generator the training set has never seen. This is where 88% is
   most likely to collapse.
2. The 710 phone photos sit in the **real class only** — `data/processed/fake`
   contains none. That is a shape/source confound: the model may have partly
   learned "non-square → real". It predicts a specific asymmetric failure, **AI
   images cropped or screenshotted to phone aspect get called Real**, which is
   exactly the bypass someone would stumble into by accident.

**Change.** Collect 100–200 AI images that are both non-square and from a
generator not in the training set. That is the only cell genuinely missing, and
it tests both problems at once. For the format axis, a ~20-line synthetic
save-path transform (random non-square crop, phone-resolution resize, chat-app
JPEG re-encode) applied to the existing holdout is enough, with perhaps 50
genuinely round-tripped images as a fidelity check.

---

## Item 6 — Backbone, only after items 1–4

**6a. Unfreeze more layers.** `train.py:192` already passes
`trainable_layers=45`; the 30 in `model.py:45` is a default no caller uses.
Revision 1 said 30 and pointed at the wrong file. **The 30 → 45 experiment has
already been run and its result was never recorded**, which lowers the expected
value of trying 60 or 80 — re-establish what 45 buys before going further.

**6b. Cheap spectral probe, before any new branch.** Compute a handful of scalar
high-frequency statistics **on the full-resolution image before the 224 resize**
— estimated noise σ, high-frequency band energy ratio, residual variance after a
3×3 median filter — and combine them with the existing CNN logit using logistic
regression on the validation split.

This tests "spectral artifacts carry signal" for a few dozen lines and no
backbone change. Because the statistics are computed pre-resize, it sidesteps the
resolution question entirely. If they add nothing, a full FFT branch is very
unlikely to pay for itself. Two liabilities: serving must keep the full-resolution
decode, which `load_image_for_inference` currently discards inside one call; and
hand-crafted noise statistics are easy to attack directly.

**6c. EfficientNet-B0 / ConvNeXt-Tiny.** Similar parameter count, stronger
features, drop-in change to `build_model`. Unmeasured.

**Conflict to respect.** Item 4 teaches the backbone to **ignore** high-frequency
energy; spectral methods **exploit** it. Run through one set of weights in that
order, 6b is evaluated on a model trained to discard the signal it needs. Feed
any spectral path the un-augmented tensor, or keep the two separate.

---

## Item 7 — Small fixes

- **The threshold is fitted on the wrong score distribution.** `train.py:219`
  fits it on `predict_probabilities`, a single forward pass, while both serving
  and `calibrate_thresholds.py` flip-average two. Cheap to fix, and worth more
  than any threshold-search machinery.
- **Too many digits.** `index.html:369` formats as `(value * 100).toFixed(2)` —
  four significant figures of a probability — for both `ai_probability` and
  `confidence`, in the single view and the batch table; `predict.py` does the
  same. Print 0 decimals or a coarse band. The raw value can stay in the JSON.
- **`confidence` is not a confidence.** `inference.py:84` rescales distance to
  threshold. It is not a probability and not a statistical confidence, yet the UI
  shows it as a headline percentage next to `ai_probability`. Two uncalibrated
  numbers derived from one uncalibrated number. Consider deleting the field —
  this is a user-facing change, so decide it deliberately rather than in passing.
- **Duplicated constants.** `inference.py:13` restates the artifact's threshold
  and band; `calibrate_thresholds.py:83` hardcodes the band again.
- **No tests.** One `test_inference.py` asserting a known-real and a known-AI
  image land on the right side of the threshold, so a preprocessing change cannot
  silently break scoring. Item 1's script can carry this as an `assert`.
- **`gemini_key.txt`.** Already in `.gitignore:6`, **untracked, and absent from
  every commit in history** — verified. Nothing in the project reads it. Delete
  the file; no rotation is needed on repository grounds. Revision 1 overstated
  this.
- **`@app.on_event("startup")` is deprecated**; move to a `lifespan` handler.

---

## Out of scope

Listed so they are explicitly declined rather than quietly dropped.

- **Frequency-domain branch.** A second input branch plus a revisited resize plus
  a new preprocessing path, which also invalidates all four measured
  non-problems at once. Item 6b is the cheap probe that tells you whether this
  would ever be worth it. Not before a deadline.
- **Platt / isotonic calibration.** Calibration is monotonic, so at the
  correspondingly mapped threshold it cannot change a single verdict. For a
  project whose threshold choice is already within noise, calibrated
  probabilities are decoration. The digit-count fix in item 7 addresses the
  actual symptom.
- **Bootstrapped threshold search.** Revision 1 proposed ~20 lines to re-derive a
  result the repo already records: `thresholds.json` states that 0.50 and 0.55
  are statistically tied. `calibrate_thresholds.py:80` already clips to
  [0.35, 0.65] and its `best < default + 0.02` guard demonstrably fired for the
  current file, so the unguarded ±0.24 swing cited in revision 1 cannot reach any
  written artifact. **The correct move is a deletion, not an addition:** freeze
  the threshold, cite the existing note, and remove the search. Item 3b still
  applies — the frozen value has to survive a retrain.

---

## What changed

### Revision 3

- **Item 0 added and fixed**: `RandomContrast` was clipping 54.5% of every
  augmented training image to black. Found by review, verified, patched.
- **Item 1 completed**: `eval_holdout.py` written and verified to reproduce the
  clean baseline, the σ=16 bypass and the JPEG non-effect.
- **Confidence intervals corrected**: revision 2's flat "±2 to ±3.5" was wrong;
  row B is ±6.0 and per-class columns reach ±8.9.
- **Item 4's acceptance tolerance corrected** from 1.5 to 3 points, because 1.5
  was below the measurement's own noise floor.
- **PNG/JPEG corpus skew recorded** as a measured non-problem, with the test that
  settles it.

### Revision 2

Errors of fact in revision 1, all verified against the code:

1. **`GaussianNoise(0.03)` was ~4× too weak.** Preprocessing happens before
   augmentation, so the constant was in [-1, 1] units: σ≈3.8 pixel units, not the
   σ=16 it was meant to defend against. The acceptance test also never checked
   σ=16.
2. **The unfrozen layer count is 45, not 30**, set at `train.py:192`. Revision 1
   pointed at a default no caller uses, so following it would have changed
   nothing.
3. **Real-world phone photos were said to need collecting.** 710 are already in
   `data/processed/real/`.
4. **`gemini_key.txt` rotation was called mandatory.** It was never committed.

Structural changes: added item 1 (the missing measurement harness) and promoted
it to first; added item 2 (free experiments that may remove the need for a
retrain); added item 3 (retrain prerequisites — 29 GB eager load, artifact
overwrite, stale baseline); inverted the threshold-bootstrap item into a deletion
and moved it out of scope; moved the frequency branch and Platt scaling to an
explicit out-of-scope list; pinned all headline numbers to one table after
finding five mutually inconsistent accuracy figures in circulation; renumbered
items into execution order and deleted the separate priority list.
