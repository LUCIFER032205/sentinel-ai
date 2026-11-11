# SENTINEL_AI — Architecture Improvement Plan

Revision 5, 2026-11-10. Revisions 1-4 were reviewed and superseded; reports in
`docs/review/`. Corrections are listed at the end under "What changed".

**Items 0, 1, 2, 2c, 3a, 3c and 4's code are done.** The item 3c baseline
retrain has run — see "Item 3c RESULT" below. It moves every number in this
document, so the pinned table now has a second block. What remains is the two
training runs and the items that depend on them. See "How to run the retrains"
below for the commands and the hazard they carry. Item 2's result removed the plan's cheap path:
both no-retrain defences were measured and both failed, so the retrain in item 4
is now unavoidable and item 3's prerequisites are unconditional.

Items are numbered in execution order. There is no separate priority list.

### Glossary

Terms used throughout, defined once here.

| Term | Meaning |
|---|---|
| `holdout_newdb` | 1,999 images (1,000 real / 999 AI) in `data/holdout_newdb/`, held out of training. The general-purpose evaluation set. |
| `holdout_hard_mined` | 600 images (300/300) in `data/holdout_hard_mined/`, selected *because* an earlier model scored them near its threshold. Adversarially hard by construction, so ~75% there is expected, not a regression. Filenames carry the mining score. |
| flip-average | Score the image and its horizontal mirror, average the two **probabilities**. This is what `inference.py` serves and what every row below uses. |
| AI-recall | Fraction of genuinely AI images called AI-generated. |
| real-spec | Real-class specificity: fraction of genuine photos called Real. |
| band | The range around the threshold (currently 0.45-0.65) in which a score is flagged `low_confidence`. It does **not** change the verdict - every upload still returns Real or AI-generated. |
| "served over HTTP" | Scored by POSTing to the running FastAPI app, exercising the real upload path. Offline rows call the model directly. Both use identical scoring; HTTP rows additionally prove the serving path is wired correctly. |
| `REPORT.md` | The project's academic write-up. It quotes older accuracy figures and currently contradicts both this document and `README.md`. |

The project is a FastAPI app (`main.py`) serving a web frontend
(`static/index.html`) plus a CLI (`predict.py`), deployed as a Docker Space on
Hugging Face.

---

## Pinned numbers

Every accuracy figure in this document refers to a row below. Nothing else is a
headline number. Rows A-D use flip-averaged scoring at threshold 0.55. **Row E is
neither** — it is a single forward pass at 0.50, written by `train.py`, which is
itself the defect item 7 records.

| ID | Dataset | Images | Threshold | Accuracy | AI-recall | Real-spec |
|----|---------|--------|-----------|----------|-----------|-----------|
| **A** | `holdout_newdb`, served over HTTP | 300 (150+150) | 0.55 | 88.0% | 84.0% | 92.0% |
| **B** | `holdout_hard_mined`, served over HTTP | 200 (100+100) | 0.55 | 75.5% | 71.0% | 80.0% |
| **C** | `holdout_newdb`, offline, grain test baseline | 600 (300+300) | 0.55 | 88.2% | 88.0% | 88.3% |
| **D** | `holdout_newdb` + `holdout_hard_mined`, offline | 1600 | 0.55 | 83.2% | 80.2% | 86.2% |
| **E** | internal train/test split, `artifacts/metrics.json` | 1000 | 0.50 | 88.8% | 89.0% | 88.7% |

Confidence intervals are 95%, in percentage points, and differ per row:
row A ±3.7, row B **±6.0**, row C ±2.6, row D ±1.8. Row E has none recorded. Per-class columns use half
the images, so they are wider still — row B's AI-recall is **±8.9**. Any
acceptance gate narrower than the relevant interval is undecidable.

These assume independent images, which holds for rows A-D: `data/holdout_newdb`
and `data/holdout_hard_mined` contain no burst sequences. The burst caveat applies
only to item 2c, which scores training-set photos.

Row E was produced by an older `train.py` than the one in the tree — its schema
does not match what `train.py` writes today (see item 3). Treat it as historical.

**All rows predate item 0's fix.** They measure the model trained while
`RandomContrast` was blanking 54.5% of every augmented image. They remain the
reference points for item 4's gate, because that gate measures a *change* and both
sides of the comparison must come from the same model. They are **not** a forecast
of what a correctly trained model scores. Item 3c re-baselines them; on the cheap
path where item 2 succeeds and no retrain happens, the current model and these rows
continue to describe each other correctly.

**`README.md` currently quotes row D's recall and specificity beside row A's
accuracy.** Both numbers are real but they come from different sets. Item 1
makes one script the sole producer of this table, and the README must be
regenerated from it in the same pass.

### Item 3c RESULT — the baseline retrain (2026-11-11)

28 epochs (10 frozen + 18 fine-tune) on all 48,707 images, `NOISE_AUG=0`, with
item 0's `RandomContrast` fix in place. Best val_auc 0.9644 at fine-tune epoch
14; `restore_best_weights` rolled back to it. Threshold re-fitted on the
validation split with flip-averaging (the item 7 fix), landing at **0.65**, up
from the hand-set 0.55.

| ID | Measurement | Old model | **New baseline** |
|----|-------------|-----------|------------------|
| **F** | `holdout_newdb` (1,999), clean | 88.2% | **90.5% ±1.3** |
| | — AI-recall | 88.0% | **95.5% ±1.3** |
| | — real-specificity | 88.3% | **85.6% ±2.2** |
| **G** | `holdout_newdb`, σ=16 grain attack | 75.3% | **83.4% ±1.6** |
| | — AI-recall under attack | 55.7% | **68.8% ±2.9** |
| **H** | internal test split (7,307), at 0.65 | — | 88.6% |

**The model got better everywhere and more robust, without any grain
augmentation.** That is item 0 plus sevenfold more data, not item 4.

**But the bypass is still real.** Grain still costs 26.7 points of AI-recall
(95.5% → 68.8%), against 32.3 points before. It is weaker, not closed.

**The operating point moved.** AI-recall is up 7.5 points while real-specificity
is down 2.7. The threshold rose to 0.65 to compensate and the model is now
recall-leaning: on `holdout_newdb` it catches 95.5% of AI images and
false-flags 14.4% of real photos. Whether that trade is right is a product
decision, not a tuning one — `thresholds.json` is the single place to change it.

### Item 4's gates, recalibrated

The old gates were written against the old baseline and are now too weak to mean
anything. σ=16 recall of 70% was a 14-point ask when the floor was 55.7%; against
68.8% it is almost a tie.

```
python eval_holdout.py data/holdout_newdb --noise 16 --check-recall 0.80
python eval_holdout.py data/holdout_newdb --check 0.875
```

- **Robustness:** σ=16 AI-recall above **80%**, up from the 68.8% this baseline
  sets. Below ~75% the grain augmentation has not earned its retrain.
- **No clean regression:** accuracy within 3 points of **90.5%**, so a floor of
  0.875. Three points because row F's interval is ±1.3 and a tighter bar could
  not be failed honestly.

---

## Where the project actually stands

| | |
|---|---|
| Model | MobileNetV2, ImageNet weights, frozen base then **last 45 layers** unfrozen (`train.py:196`) |
| Input | 224×224, aspect squashed, preprocessed to [-1, 1] at load time (`data.py:26`) |
| Scoring | flip-average, 2 forward passes, single threshold from `artifacts/thresholds.json` |
| Training corpus | 24,763 real + 23,944 fake = **48,707 images** in `data/processed/` |
| Known bypass | added grain, σ=16 → AI-recall falls to 68.8% (was 55.7% pre-retrain) |

### Measured non-problems — do not spend effort here

- JPEG re-compression to q30, downscaling to 0.25×, screenshot-style re-encoding:
  all within noise.
- Inference resolution: 320 neutral at 2× cost, 448 **−4.3 pts** at 4× cost.
- Aspect handling: centre-crop and letterbox both tie with squashing on 4:3, 16:9
  and 9:16. Squashing is fine.
- Rotation test-time augmentation: 8-variant averaging scores below flip-average
  at 4× the cost.
- **File format is not a measurable cue, despite a skewed corpus.** `data/processed/real` is
  0 PNG out of 24,763; `data/processed/fake` is 7,903 PNG out of 23,944, and the
  holdout repeats the skew (0/1,000 vs 267/999). A PNG carries no JPEG artifacts,
  so "no artifacts → fake" is learnable in principle. Measured: PNG fakes score
  0.858 mean p(AI) against 0.848 for JPEG fakes, a difference of **+0.010, CI
  [-0.049, +0.060]** — indistinguishable from noise, though that interval still
  admits an effect of up to ~0.06, so this is "not detectable at this sample size",
  not "proven absent". The `--jpeg 30` run in item 1 agrees, since it converts PNG
  fakes to JPEG and barely moves. Worth fixing as
  dataset hygiene before the next corpus expansion; **not** a reason to distrust
  rows A-D.

These were measured on the current single-branch 224×224 spatial model. A
frequency-domain branch would invalidate all four at once — which is one reason
it is out of scope (see the end).

---

## Item 0 — DONE: training augmentation was destroying images

`RandomContrast(0.1)` at `train.py:56` takes `value_range=(0, 255)` by default and
**clips to that range**. Tensors arriving from `data.py:26` are in [-1, 1], so every
value below mid-grey was clamped to 0.0. Measured: **54.5% of each augmented image
was blanked to black**, and no negative value survived.

Fixed by passing `value_range=(-1, 1)`. Verified: the output range is restored to
[-1, 1], negatives survive, and contrast still varies across draws.

This is not a units subtlety like the one in item 4 — it is destructive clipping,
and it affected every training run that produced the current model, including the
one behind every number in the pinned table. `RandomZoom` and `RandomRotation` do
not have this fault.

**Live consequence:** item 3c's re-baseline is the first run that will benefit, and
until it happens nobody knows what this fix is worth. It could move accuracy on its
own.

---

## Item 1 — DONE (with follow-ups listed at the end of this section)

`eval_holdout.py` is committed. It is the only code in the tree that reads
`data/holdout_*`.

```
python eval_holdout.py data/holdout_newdb [--noise S] [--jpeg Q] [--scale F]
                       [--limit N] [--seed N] [--threshold T]
                       [--check MIN] [--check-recall MIN]
```

It reuses `data.iter_image_files` and `data.preprocess_image`, reads the live
threshold through `inference.load_calibration`, and flip-averages exactly as
`inference.py` does. Perturbations are applied to the decoded 0-255 BGR image
**before** preprocessing and before the 224 resize, matching how the published
numbers were measured.

`--check` gates **accuracy**; `--check-recall` gates **AI-recall**. Both can be
given at once and either failing exits non-zero. Robustness gates need
`--check-recall`, because noise pushes AI images toward "Real" while
real-specificity *rises* — accuracy alone understates the damage.

This covers the regression-test half of item 7's "no tests" bullet at whole-set
granularity. It is not the two-image sanity assert that bullet also describes;
that remains open.

**Verified against prior measurements** (300 images per class, seed 0):

| Run | Result | Expected |
|---|---|---|
| clean | 88.3% ±2.6 | **row C**, 88.2% ±2.6 (same size, same offline path) |
| `--noise 16` | AI-recall **55.7%** ±5.6 | 55.7% from the scratchpad run |
| `--jpeg 30` | 88.0% ±2.6 | non-effect |

The σ=16 reproduction is exact, so the acceptance test for item 4 now runs.

**Why this item existed.** Until this was written, every published number came
from throwaway scripts in a temporary directory outside the repository, unversioned
and about to be lost. No committed code read `data/holdout_*` at all — the only
evaluation ran on the internal train/test split — so the 2,599 holdout images on
disk were unreachable, and revision 1 cited an acceptance test of "re-run
`scratchpad/grain.py`", a path that never existed here. That is now resolved.

Note: `calibrate_thresholds.py:30` `predict_paths` was *not* reused. It calls
`load_image_for_inference`, which decodes, resizes and preprocesses in one call and
leaves no seam to inject a perturbation at the right stage. `eval_holdout.py` has
its own loop for that reason.

**Still to do here:** regenerate the pinned-numbers table and `README.md` from this
script so there is one producer of those figures, and reconcile `REPORT.md`, which
contradicts both.

---

## Item 2 — DONE: the cheap fixes were tried and both failed

`eval_holdout.py` gained `--score-noise` (grain-floor normalisation) and
`--tta-noise` (noise-level TTA) so these could actually be run. Measured on 200
images per class from `holdout_newdb`, seed 0:

| Configuration | Accuracy | AI-recall |
|---|---|---|
| clean, no defence | 88.2% ±3.2 | 82.5% ±5.3 |
| σ=16 attack, no defence | 77.2% ±4.1 | 56.5% ±6.9 |
| σ=16 attack + grain floor σ=8 | 75.8% ±4.2 | **52.5%** ±6.9 |
| σ=16 attack + noise-TTA σ=8 | 77.5% ±4.1 | **56.0%** ±6.9 |
| σ=16 attack + grain floor σ=16 | 71.0% ±4.4 | **43.5%** ±6.9 |
| σ=16 attack + noise-TTA σ=16 | 74.8% ±4.3 | **50.5%** ±6.9 |

Neither defence recovers any recall at any setting tried, and both *cost*
accuracy on clean input — the grain floor drops it from 88.2% to 81.2%, TTA to
85.2%.

**Why the theory failed.** The premise was that the model reads a *relative* noise
floor, so equalising it would destroy the cue. The model's response to noise turns
out to be **monotonic, not differential**: adding grain pushes everything toward
"Real" regardless of class. Real-specificity climbs to 99% while recall collapses.
You cannot equalise a cue by adding noise to a model that has no noise invariance —
only training can give it that.

**Consequence: there is no cheap path.** Item 4's retrain is necessary, item 3's
prerequisites are unconditional, and the branch this plan was built around is gone.

**2c. Score the 710 phone photos already in the corpus.** `data/processed/real/`
contains 710 `phone_real_IMG_*.jpg` files: genuine phone photos, non-square,
non-1024. Revision 1 claimed real-world data needed collecting. It does not, for
the real class. The number is optimistic because these files are in training, so
report it as a ceiling, not an estimate.

Caveat: these are not 710 independent scenes — `Burst01`…`Burst16` are
near-duplicates of a single moment. Report distinct scenes, not file count, and
group-split by scene if they are ever held out.

**2c RESULT.** `eval_holdout.py` gained `--filter`; `--filter phone_real_IMG`
over `data/processed` scores the 710 phone photos on their own:

| | |
|---|---|
| real-specificity | **99.2% ±0.7** |
| for comparison, held-out real images | 92-94% |

**Read this as a ceiling, not an estimate.** All 710 are in the training set, and
the ~7-point gap over held-out real photos is the memorisation showing. It does
not say the detector handles phone photos well; it says it has seen these ones.

Scene caveat, as promised: 16 of the 710 are a single burst
(`IMG_20191221_104738_Burst01..16`), so the effective count is roughly 695 scenes
across 32 capture days — a smaller correction than feared. They remain one
person's camera roll, not a sample of phone photography in general.

**Still to do here:** Reproduce with:

```
python eval_holdout.py data/holdout_newdb --noise 16 --score-noise 8
python eval_holdout.py data/holdout_newdb --noise 16 --tta-noise 8
```

**The floor item 4 must beat is therefore 56.5% σ=16 recall** — the undefended
number — not anything these two produced.

---

## Item 3 — Retrain prerequisites

Item 2 established there is no cheap path, so these are **unconditional**, not
contingent. Skipping any of them makes the retrain's result uninterpretable.

Do 3b and 3b-bis **first and today**, before any further measurement: they protect
an artifact that nothing in the repo can regenerate, and every remaining item in
this plan touches it.

**3a. `load_dataset` cannot load this corpus.** `data.py:29` decodes and
preprocesses every image into one resident float32 array: 48,707 × 224×224×3 × 4
bytes ≈ **29 GB** before batching. Per-epoch decode will slow epochs on a
disk-bound machine; `.cache()` to a local file recovers most of it.

Three options, **not interchangeable**:

1. **`tf.data` with map-decode, keeping `preprocess_image`.** Preferred. The
   normalisation stays where it is, so item 4's σ table stays valid.
2. **Pin a fixed subset.** Safe, loses data.
3. **`tf.keras.utils.image_dataset_from_directory`.** ⚠ It does **not** apply
   `preprocess_input`, so tensors arrive in [0, 255] instead of [-1, 1]. Every σ in
   item 4 would then be ~127× too weak, and the augmenter's `value_range=(-1, 1)`
   from item 0 would be wrong again in the opposite direction. If you take this
   route you must move normalisation explicitly and re-derive the σ table. The
   failure is silent and shows up only after a multi-day retrain.

**3b. Snapshot the artifacts first.** `train.py:225` overwrites
`artifacts/thresholds.json` and `train.py:228` overwrites
`artifacts/metrics.json`. Both are currently hand-shaped:

- `thresholds.json` holds threshold 0.55 and a `tuning_note` that
  `find_best_threshold` never writes. **The served threshold is a value no code
  in the repo can regenerate.** The first retrain silently reverts it.
- `metrics.json` has a flat schema; `train.py:139` writes a nested one. The
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
1,000, which under the 70/15/15 split at `train.py:32-45` implies a **corpus** of
roughly 6,700 images for that run — a training set of about 4,700. Today's corpus is
48,707, around sevenfold larger. Without a clean re-baseline, any clean-accuracy
floor (revision 2 proposed 87%; that figure is retired and replaced by the relative
gate in item 4) compares against a model trained on a different dataset and means
nothing. **The retrain
runs twice: once without noise augmentation, once with.**

---

## How to run the retrains

Measured throughput on this machine: **0.43 s per batch of 32** in the frozen
phase, 1,066 steps per epoch over the 34,095 training images. The default
schedule (10 frozen + 18 fine-tune epochs) is about **5 hours per run, 10 hours
for both**. CPU only — TensorFlow has no GPU support on native Windows.

**Hazard: `train.py` overwrites the live model mid-run.** `ModelCheckpoint` at
`train.py:256` and `:283` writes straight to `artifacts/ai_image_detector.keras`,
and the run then overwrites `thresholds.json` and `metrics.json` at the end. The
server will be loading a half-trained model while a run is in progress.

A pre-retrain snapshot is at `artifacts/snapshot_pre_retrain/`. The two JSON
files are also tracked in git as of item 3b, so they can be restored with
`git checkout -- artifacts/thresholds.json artifacts/metrics.json`. The model
file is not in that snapshot's git history — the copy on disk is the only one.

Run the baseline first. It differs from the grain run **only** in `NOISE_AUG`:

```bash
NOISE_AUG=0 python train.py    # item 3c: re-baseline on today's 48,707 images
```

```bash
NOISE_AUG=1 python train.py    # item 4: the same run, with grain
```

Then gate the result:

```bash
python eval_holdout.py data/holdout_newdb --noise 16 --check-recall 0.70
```

```bash
python eval_holdout.py data/holdout_newdb --check <baseline accuracy minus 0.03>
```

The baseline run exists to supply that second number. Until it finishes, there is
no honest clean-accuracy floor to compare against: the current pinned table was
measured on a model trained while `RandomContrast` was blanking 54.5% of every
augmented image, and on a corpus a seventh the current size.

---

## Item 4 — Close the grain bypass

**Evidence.** Measured on row C's images. Monotonic and significant from σ=4:

| σ (0–255 units) | Accuracy | AI-recall | vs clean |
|---|---|---|---|
| 0 | 88.2% | 88.0% | — |
| 4 | 85.2% | 80.0% | −3.0, significant |
| 8 | 82.5% | 72.3% | −5.7, significant |
| 16 | 75.3% | **55.7%** | −12.9, significant |

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

**DONE (code only — the run has not happened).** Grain is added in a `tf.data`
map, not in the augmenter `Sequential`: per-example σ and a batch fraction are not
expressible in a `Sequential`. σ ~ U(0, 16) pixel units per example, on 50% of
each batch. Verified that 50.4% of a batch receives grain, that realised σ spans
0 to ~17.8 pixel units and reaches all three measured rungs, and that output stays
clipped to [-1, 1].

`NOISE_AUG=0` disables it for the baseline run. It is read with `os.getenv`, not
`get_env_int` — that helper returns its default for any non-positive value, so
`NOISE_AUG=0` meant "on" and both retrains would have been identical.

`RandomContrast` was the only layer with a value-range hazard and item 0 fixed it.
`RandomZoom` and `RandomRotation` are range-agnostic and need no change. Re-check
this only if item 3a moves where normalisation happens.

**Acceptance.** Two gates, both via item 1's script. Run the full
`holdout_newdb` (no `--limit`) so the interval is as tight as the data allows:

```
python eval_holdout.py data/holdout_newdb --noise 16 --check-recall 0.70
python eval_holdout.py data/holdout_newdb --check <re-baseline minus 0.03>
```

1. **Robustness:** σ=16 AI-recall above 70%, up from 55.7%. This needs
   `--check-recall`, not `--check`: under noise, real-specificity *rises* to 98%
   while recall collapses, so accuracy alone hides the failure. Verified against the
   current model, this gate fails as it should — AI-recall 55.3%, below 0.700.
2. **No clean regression:** accuracy within **3 points** of the item 3c
   re-baseline. The floor is computed from that run, not hardcoded here, because
   3c has not happened yet; writing a literal now would reintroduce the stale
   baseline 3c exists to remove. If item 2 dissolved the retrain and 3c never ran,
   use row C's 88.2%, making the floor 0.852.

3 points, not revision 2's 1.5, because 1.5 sits inside row C's own ±2.6 interval —
a gate that cannot be failed honestly is not a gate.

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

**Prerequisite: establish what generators are already in the corpus.** Nothing in
the repository records the provenance of `data/processed/fake`. `artifacts/` holds
`gemini_prompt_bank_500.csv` and `gemini_prompting_playbook.md`, and
`data/raw/gemini_augmented`, `gemini_hard_augmented`, `gemini_hard_misses` and
`gemini_test` all exist, which points at Gemini as at least one source — but that is
inference from folder names, not a record. **Write the provenance down before
collecting anything**, or "a generator not in the training set" cannot be satisfied
or verified.

**Change.** Collect 100–200 AI images that are both non-square and from a generator
confirmed absent from that record. That is the only cell genuinely missing, and it
tests both problems at once.

For the format axis, a ~20-line synthetic save-path transform applied to the
existing holdout is enough. Reuse the settings already measured so the result is
comparable: random non-square crop to 4:3 or 16:9, resize to a phone resolution
(1080×1440 or 1080×1920), JPEG re-encode at q70. Keep perhaps 50 genuinely
round-tripped images — sent through a real chat app and saved back — as a check
that the synthetic transform resembles the real save path.

---

## Item 6 — Backbone, only after items 1–4

**6a. Unfreeze more layers.** `train.py:196` already passes
`trainable_layers=45`; the 30 in `model.py:45` is a default no caller uses.
Revision 1 said 30 and pointed at the wrong file. **The 30 → 45 experiment has
already been run and its result was never recorded**, which lowers the expected
value of trying 60 or 80.

Re-establishing what 45 buys means a 30-layer comparison run, which is a retrain and
therefore inherits all of item 3's prerequisites. There is no cheap version. If item
2 dissolved the retrain, this item is not cheap either and should be weighed against
item 6b, which needs no training at all.

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
energy; spectral methods **exploit** it. *If* both are run through one set of
weights in the order this plan gives — item 4 before item 6b — *then* 6b is
evaluated on a model deliberately trained to discard the signal it needs, and a
negative result would say nothing about whether spectral signal exists.

Avoid it one of two ways: feed the spectral path the un-augmented tensor while the
spatial path gets the noise, or keep the two as separate models. Note that 6b as
specified computes its statistics on the raw image before preprocessing, so it
sidesteps this automatically — the conflict only binds if 6b is later promoted into
a branch sharing the backbone's weights.

---

## Item 7 — Small fixes

- **The threshold is fitted on the wrong score distribution — but only fix this
  if the search survives.** `train.py:223` fits it on `predict_probabilities`, a
  single forward pass, while both serving and `calibrate_thresholds.py`
  flip-average two. This matters only while `find_best_threshold` still chooses
  the served value. The out-of-scope section recommends deleting that search and
  freezing the threshold instead, which removes the defect rather than repairing
  it. **Pick one:** delete the search (preferred, smaller) *or* fix the fit. Doing
  both is contradictory, and doing neither leaves `train.py` silently replacing a
  hand-set threshold with a badly-fitted one on the next run — which is what
  item 3b describes.
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
  [0.35, 0.65] and its `best < default + 0.02` guard at `:75` demonstrably fired
  for the current file, so the unguarded ±0.24 swing cited in revision 1 cannot reach any
  written artifact. **The correct move is a deletion, not an addition:** freeze
  the threshold, cite the existing note, and remove the search. Item 3b still
  applies — the frozen value has to survive a retrain, and today it is not even
  tracked by git (3b-bis).

  Taking this option retires item 7's threshold-fit bullet, since there is no fit
  left to correct. Those two entries are alternatives, not a sequence.

---

## What changed

### Revision 5

Stage 7 (design / structural / style critics) raised 3 Blockers, all of them
re-raises of earlier findings that previous revisions had answered in prose rather
than in code. Acting on the first one settled the plan's central question.

- **Item 2 was unrunnable.** Its decision-rule commands applied `--noise` once, as
  an attack. 2a and 2b are *defences* — a second noise stage and a second scoring
  pass — and `eval_holdout.py` could express neither, so both commands would have
  measured the unmodified system and reported it as a result. Fixed by adding
  `--score-noise` and `--tta-noise`, then running them: **both defences failed**,
  which deleted the cheap path and with it most of this document's branching.
- **Item 3's prerequisites were gated behind the branch that did not need them.**
  On the cheap path, 2a's threshold re-measurement would have overwritten the
  untracked `thresholds.json` and destroyed the `tuning_note` the out-of-scope
  section tells the reader to cite. Now unconditional, and flagged do-this-first.
- **Item 3a offered three loaders as equivalents.** `image_dataset_from_directory`
  does not apply `preprocess_input`, which would invert item 4's entire σ table and
  train the noise ~127× too weak — discoverable only after a multi-day retrain.
  Now marked, with the preferred option named.
- **Every `train.py` citation at or after line 52 was off by +4**, caused by item
  0's own four-line comment. Corrected against the file.
- **The burst-sequence caveat was attached to the wrong data.** No `Burst*` files
  exist in `data/holdout_newdb`; they are in `data/processed/real`, so the caveat
  belongs to item 2c alone and rows A-D are unaffected.
- Corrected: item 4's block told the reader to check `RandomZoom` for a hazard the
  same document says it does not have; item 0 was still scheduling itself as future
  work; "File format is not a cue" was a bolded negative existence claim on an
  interval that admits an effect up to 0.06.

### Revision 4

Found by the Stage 6 comprehension test, which read this document alone with no
codebase access — so every item below is a place the document failed to stand on
its own, not a disagreement about the work.

- **Item 1 contradicted itself**: it announced the harness as committed, then four
  paragraphs later stated in the present tense that no committed code reads the
  holdout sets. The motivation paragraph is now past-tensed.
- **Item 4's acceptance commands did not test item 4's acceptance criteria.** They
  gated on accuracy where the criterion is AI-recall, with numbers matching neither
  reading, and one literal floor that pre-empted the very re-baseline it depended
  on. `eval_holdout.py` gained `--check-recall` because the criterion was not
  expressible before.
- **Rows A-D were not marked as pre-dating item 0's fix**, so the table read as a
  forecast for a model that no longer exists.
- **Item 7 and the out-of-scope list gave opposite instructions** about the
  threshold search. They are now stated as alternatives.
- **"clean accuracy must not fall below 87%"** was quoted as though it existed
  somewhere; it was revision 2's own wording. Retired explicitly.
- **Item 5 was unactionable**: it asked for a generator absent from the training
  set without the document ever recording which generators are in it.
- **Item 6's "conflict to respect" was missing a conditional** and could be read as
  an instruction to do the harmful thing.
- **Item 2's decision rule had no number**; it now uses item 4's own bar.
- Corrected: the verification table compared an offline 600-image run against row A
  (300, HTTP) when it matches row C; "a training run on roughly 6,700 images" should
  have read "a corpus of roughly 6,700"; holdout total is 2,599, not 2,257; row E is
  single-pass at 0.50, not flip-averaged; interval units and confidence level were
  never stated; "real" in the σ table meant "statistically significant" beside a
  class named Real.
- Added a glossary. The reviewer could not determine what `holdout_newdb`,
  `holdout_hard_mined`, flip-average, real-spec, the band, `REPORT.md` or "served
  over HTTP" meant, and the document never mentioned that the project is a deployed
  web app.

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
2. **The unfrozen layer count is 45, not 30**, set at `train.py:196`. Revision 1
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
