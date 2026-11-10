# Goldfish Review Report — Stage 6 (Comprehension Test)

**Target:** `E:\ML mini-project\docs\architecture-improvement-plan.md` (full file, revision 3)

**Reviewer:** `egm-comprehension`, reading the document alone — no codebase access, no
sight of the Stage 5 reports.

**Standard applied:** none. The test is self-containment, not detail.

The two paraphrase sections below are what a fresh reader actually took away. Where they
diverge from intent, the document is at fault, not the reader.

---

## What this proposes

The plan orders eight numbered work items for an AI-vs-real image classifier, with items 0
and 1 already done: fix a training augmentation that was clipping half of every augmented
image to black, and commit `eval_holdout.py` as the one script that reads the holdout sets
so that measurements stop living in throwaway scratch files. The remaining work is sequenced
around one real defect — adding Gaussian grain at σ=16 (0–255 units) drops AI-recall from
88% to 55.7% — and the plan deliberately tries the cheap fixes first: three no-retrain
experiments (fixed grain injected into every input, noise-level test-time averaging, and
scoring 710 phone photos already in the corpus) which, if they recover most of the lost
recall, demote the fix from a retrain to a preprocessing change. If a retrain is still
needed, three prerequisites are declared blocking (the eager loader needs ~29 GB, the
hand-shaped `artifacts/*.json` files get silently overwritten and the served threshold is
untracked, and the existing baseline was trained on a corpus a seventh the current size),
then noise augmentation is added with per-example σ sampled over the measured attack range.
Everything else is secondary: collect 100–200 non-square AI images from an unseen generator
(the one genuinely missing data cell), a cheap pre-resize spectral probe before any new
backbone branch, and a list of small fixes. A frequency-domain branch, Platt/isotonic
calibration, and a bootstrapped threshold search are explicitly declined, the last one
inverted into a deletion.

## How the existing system works, per this document

A MobileNetV2 with ImageNet weights, last 45 layers unfrozen, takes 224×224 aspect-squashed
input preprocessed to [-1, 1] at load time; scoring runs two forward passes (an image and
its flip), averages them, and compares against a single threshold of 0.55 read from
`artifacts/thresholds.json`, with a ±0.10 band. It was trained on 48,707 images in
`data/processed/`, and sits at roughly 88% accuracy on clean holdout data but 75.5% on a
hard-mined holdout, with the grain bypass above as its known failure. Measurement is
scattered: five mutually inconsistent accuracy figures were in circulation, so this revision
pins every number to one table of rows A–E, and `README.md` still mixes row D's recall with
row A's accuracy while `REPORT.md` contradicts both. The threshold artifact and metrics file
are hand-written, untracked by git, and reproduce in deployment only because a fallback
constant happens to hold the same value; the metrics file's schema does not match what the
current `train.py` writes, so the recorded baseline is historical.

---

## Findings

### 1. Blocker — Item 1 contradicts its own heading; status is unreadable

`Item 1 — DONE: measurement harness is in the repository` opens with "`eval_holdout.py` is
committed. It is the only code in the tree that reads `data/holdout_*`." Four paragraphs
later, under bold **Problem.**, the same section states "Every number above was produced by
throwaway scripts in a temporary directory outside the repository... **No code in the tree
reads `data/holdout_*` at all**... **The 2,257 holdout images on disk are currently
unreachable by any committed code.**" These are direct negations of each other, both in
present tense. A fresh reader cannot tell whether the harness exists. The **Problem**
paragraph is probably un-past-tensed revision-2 text, but the document gives no signal — it
is not marked "was", "previously", or struck. The same section then lists "**Still to do
here:**...", so the item is labelled DONE while carrying open work. Either past-tense the
Problem paragraph and retitle to "partially done", or split the leftovers into their own item.

### 2. Blocker — The item 4 acceptance commands do not encode the item 4 acceptance criteria

Item 4 states the gate in prose: "σ=16 AI-recall rises above 70% (from 55.7%), **and** clean
accuracy stays within **3 points** of the item 3c re-baseline." The commands given are:

```
python eval_holdout.py data/holdout_newdb --noise 16 --check 0.85
python eval_holdout.py data/holdout_newdb --check 0.86
```

Three mismatches, none resolvable from the document:

- **What `--check` compares is never stated.** Item 1 says only "exits non-zero below a
  floor". Floor of *what*? The gate needs AI-recall for the first command and accuracy for
  the second, which one flag cannot do unless it changes meaning with `--noise`.
- **The numbers match neither reading.** If `--check` is accuracy, `--noise 16 --check 0.85`
  demands 85% where measured σ=16 accuracy is 75.3%. If it is AI-recall, 0.85 is not 0.70.
- **`--check 0.86` cannot express "within 3 points of the 3c re-baseline"**, because 3c has
  not run. 0.86 reads as a hardcoded guess from row C's 88.2%, reintroducing the stale
  baseline comparison item 3c exists to eliminate.

Also: acceptance is scoped "on the row C images" (600), but the commands pass no `--limit`,
so they run the full `holdout_newdb`. The stated ±2.6 interval does not apply to them.

### 3. Major — Item 0's fix invalidates the model behind the pinned table, and the document never says so

Item 0 says the clipping bug "affected every training run that produced the current model".
Rows A–D, item 4's σ table, and item 1's verification runs were all measured on that model.
The document never states whether those numbers survive the fix, need re-measuring, or are
knowingly retained as pre-fix references. This matters because item 4's gate is defined
relative to them (55.7% → >70%). On the cheap-fix path (item 2 succeeds, item 3c never runs)
nothing ever re-measures the post-fix model. State the intent explicitly.

### 4. Major — Item 7 and "Out of scope" give opposite instructions for the threshold

Item 7 says fix `train.py:219`, which fits the threshold on a single forward pass while
serving flip-averages. Out of scope says "freeze the threshold... and remove the search". If
the search is removed there is no fit to correct; if the fit is corrected the search stays.
Item 3b additionally says the served 0.55 "is a value no code in the repo can regenerate",
which implies the existing fit is already not the source of the served threshold — making
item 7's fix pointless. Three statements needing one reconciling sentence.

### 5. Major — "clean accuracy must not fall below 87%" is quoted as an existing gate but 87% appears nowhere else

Item 3c quotes it with quotation marks, implying it exists somewhere, but the document never
introduces 87%, never says where it is written, and never says whether it is retired. Item 4
uses a different formulation. A reader cannot tell if 87% still binds anything.

### 6. Major — Item 5's deliverable is not actionable from the document

"Collect 100–200 AI images... from a generator not in the training set." The document never
names the generators the training set uses — only "one homogeneous corpus". Without that,
the criterion cannot be satisfied or verified. The "~20-line synthetic save-path transform"
lists three operations but no parameters (which phone resolutions, which JPEG quality),
while the non-problems section already measured q30 and 0.25×; unclear whether to reuse those.

### 7. Major — The "Conflict to respect" sentence in item 6 is not parseable

"Run through one set of weights in that order, 6b is evaluated on a model trained to discard
the signal it needs." Missing a conditional. It reads either as a warning ("*If* you run
both through one set of weights...") or as an instruction to run them in that order. This
sentence governs whether items 4 and 6b can share a training run.

### 8. Major — The decision rule in item 2 has no number

"If 2a or 2b recovers most of σ=16 recall" — "most" is undefined, and "report the result as
the floor a retrain has to beat" gives no figure. Item 4's gate is a specific >70%; unclear
whether 2a/2b are judged against the same bar. Also unstated: whether 2a and 2b can be
combined, and whether 2b requires re-measuring the threshold as 2a does. 2a's "re-measure
both" does not say both *what*.

### 9. Major — `--check`'s metric, and the smoke-test claim that rests on it

Item 1 claims `--check` "makes it the smoke test item 7 asked for"; item 7 wants a test that
a known-real and a known-AI image land on the right side of the threshold. A whole-set
accuracy floor and a two-image sanity assert are different tests. The document asserts one
satisfies the other without explaining how.

### 10. Minor — Item 1's verification table compares an offline 600-image run against row A

The table header says "(300 images per class, seed 0)" — 600 images — with a clean result of
"88.3% ±2.6", but the Expected column cites "row A, 88.0% ±3.7". Row A is 300 images total,
**served over HTTP**, ±3.7; row C is 600 images, **offline**, 88.2% ±2.6. The run matches
row C in size, interval and offline-ness but is labelled row A. The document also never
explains what "served over HTTP" changes about a score, so a reader cannot judge whether an
offline run reproduces an HTTP row at all.

### 11. Minor — Confidence intervals: no units, no level, one row missing

"row A ±3.7, row B ±6.0, row C ±2.6, row D ±1.8" — units never stated (percentage points
assumed), confidence level never stated (95% assumed). Row E has no interval, yet row E is
the only row at threshold 0.50 and underpins the re-baseline discussion. The header says all
rows are flip-averaged "unless the row says otherwise"; row E says 0.50 but nothing about
flip-averaging, and item 7 states `train.py` fits on a single pass, so row E is probably not
flip-averaged. The escape clause does not cover this.

### 12. Minor — Undefined project-specific terms and identifiers

None of these are defined anywhere in the document:

- **`holdout_newdb`** / **`holdout_hard_mined`** — what "newdb" is; what makes the hard set
  hard, or how it was mined. It is 12.5 points worse, so its construction is load-bearing.
- **`Real-spec`** (table column) — never expanded.
- **flip-average** — which flip (horizontal assumed), and whether the average is of
  probabilities or logits.
- **band** / "0.45-0.65 band" — what the band *does* to a verdict is never said, yet 3b-bis
  treats divergence in it as a deployment hazard.
- **"non-1024"** (item 2c) — only meaningful if the rest of the corpus is 1024×1024.
- **Hugging Face Space** — the deployment target appears once, with no prior mention that
  the project is deployed. Likewise `index.html`, `predict.py` and `@app.on_event` imply a
  web app and FastAPI server never introduced in "Where the project actually stands".
- **`REPORT.md`** — named as contradicting the table and README, never described.
- **±0.24 swing** — cited from revision 1, which is not in this document.
- **`tuning_note`** — its content is only alluded to, yet it is the sole justification for
  freezing the threshold.

### 13. Minor — Item 3c's arithmetic wording is off by one step

"test support of 1,000, which under the 70/15/15 split implies a training run on roughly
6,700 images". 1,000 / 0.15 ≈ 6,700 is the **whole corpus**; the training portion is ~4,700.
The sevenfold comparison that follows is corpus-to-corpus and correct, so the wording
misstates the training-set size. The 70/15/15 split is also asserted with no file reference,
unlike nearly every other claim.

### 14. Nit — Holdout set sizes are only recoverable by arithmetic

The document never states how many images each holdout contains; it is reconstructible from
the PNG bullet and item 1's "2,257". Rows A (300), B (200), C (600) and D (1,600) draw four
different subset sizes with no stated reason.

### 15. Nit — Underspecified parameters in the item 4 change

"sampling σ per example across the measured attack range... applying it to a fraction of
each batch" — sampling distribution and fraction both unspecified, and the fraction is the
parameter trading robustness against the clean-accuracy gate in the same section. Same for
2a's "σ≈8", where every other σ is exact.

### 16. Nit — The word "real" is overloaded in item 4's table

The `vs clean` column reads "−3.0, real", meaning statistically real. "Real" is also a class
name, and the adjacent column is AI-recall. Use "significant".

### 17. Nit — Backward reference to a forward-defined flag

The non-problems section says "The `--jpeg 30` row above agrees" — `--jpeg` is introduced in
item 1, below, and the thing "above" is a bullet, not a row.

### 18. Nit — Item 6a's follow-up has no procedure

"re-establish what 45 buys before going further" requires a 30-layer comparison run, i.e. a
retrain, which item 3 gates and item 2 tries to avoid. The document does not say whether 6a
inherits item 3's prerequisites. Item 6b's "logistic regression on the validation split"
does not define the validation split; the only split mentioned belongs to the stale
historical run, and item 3a proposes replacing the loader.

---

## Verified as correct by the reviewer

No findings against the "Out of scope" rationales, the unit-conversion table (σ_pixel /
127.5 checks out at 4, 8 and 16, and the σ≈3.8 back-calculation of `GaussianNoise(0.03)` is
correct), the 29 GB estimate (48,707 × 150,528 × 4 bytes ≈ 29.3 GB), or the changelog —
described as the clearest part of the document.
