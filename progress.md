## 2026-09-20 · Environment
- Env: Python 3.11.9, torch 2.14+cpu, CLIP (1,512), MiniLM (1,384). Smoke test OK.
- Decision: CPU-only torch — no GPU needed for prep-stage experiments.
- Decision: dataset generated not collected → data/ gitignored, seed + generator are the artifact.


## 2026-10-04 · Days 2–4
**Done**
- Transform spec written (spec/transform_spec.md)
- build_affine implemented; matches cv2.getRotationMatrix2D exactly
- 6 unit tests pass: identity, rotation sign, centre fixed, translation units, round trip, OpenCV agreement
- GeoBench 2D downloaded (5,677 examples) into data/hf_cache
- Dataset vocabulary and sentence templates (vocab.py)

**Decisions (with reasons)**
- Positive θ = counter-clockwise on screen: matches OpenCV, which does the actual warping
- build_affine recomputes θ from cos/sin via atan2, so model outputs that aren't exactly unit-length still work
- Translation stored as a fraction of width so the model is resolution-independent; converted to pixels inside build_affine (named tx_px so units stay visible)
- pytest limited to tests/: it was picking up tools/smoke_test.py, whose name ends in _test.py
- Paging file added on D: (8–16 GB) after CLIP failed to load with OSError 1455; C: has only ~11 GB free
- GeoBench adopted as external benchmark and as the source of images and masks, replacing COCO: it provides real ground-truth target masks
- GeoBench conversions: θ = −rz (its clockwise is positive), translation = pixels / 512
- Training vocabulary excludes GeoBench's distinctive words, so GeoBench tests wording the model has never seen
- GeoBench cached under data/ — on D:, and already ignored by git

**Issues**
- Earlier assumption that no dataset pairs instructions with geometric parameters was wrong — GeoBench does.

**Next**
- Measure GeoBench's ranges per adverb, then set the level ranges in distributions.py


## 2026-10-05 · GeoBench measured
**Done**
- tools/geobench_stats.py: measured all 5,677 GeoBench 2D rows

**Findings**
- Parameter format confirmed as [tx, ty, tz, rx, ry, rz, sx, sy, sz]: 0 rows violate it
- Rotation sign confirmed opposite to ours: all 785 counterclockwise rows have rz < 0, so θ = −rz
- Single edits only: 2,635 scale, 1,603 rotate, 1,439 translate, no combined edits
- Exactly 3 magnitude levels, identical across edit types:
  small (gently, mildly, lightly, slightly), medium (appreciably, markedly, moderately),
  large (heavily, strongly, significantly, intensely)
- Ranges per level — translate (fraction of width): 0.05–0.13 / 0.10–0.27 / 0.20–0.47;
  rotate: 5–9° / 10–19° / 20–39°; |log s|: 0.10–0.26 / 0.23–0.51 / 0.41–1.06
- Translation and scale ranges overlap between levels because GeoBench scales shifts to object size;
  rotation levels do not overlap
- Our original planned ranges were well off (e.g. "significantly" rotation planned 40–70°, real 20–39°)


## 2026-10-06 · Magnitude ranges
- distributions.py: one range per edit type and level, set from GeoBench's measured min/max
- Decision: 3 levels (small/medium/large) to match GeoBench; dropped our "huge" level, which had no real data to calibrate against; "drastically" moved to large, "massively" removed
- Decision: uniform sampling within each range, from one seeded random generator passed in, so the dataset is reproducible


## 2026-10-06 · Generator v1
- generate.py: single-transform examples (translate / rotate / scale in equal shares), vague magnitude words only
- First run: 5,000 examples, seed 42, into data/dataset_v1.jsonl (regenerated from code; not in git)
- Decision: held out the verbs "slide" and "expand", alongside the held-out adverbs, to test unseen verbs
- Decision: negation dropped from scope; combined edits and explicit numbers come next
- Decision: start with 5,000 examples so the output can be checked by eye; scale up once correct
- Decision: Python's built-in random instead of numpy's — no type conversions needed for JSON
- Fix: rotation direction now chosen 50/50 before choosing a phrase; picking uniformly among phrases made 2/3 of rotations counter-clockwise, because two of the three phrases mean counter-clockwise


## 2026-10-09 · Whole-image edits
- Decision: edits apply to the whole image, with rotation and scale about the image centre (W/2, H/2), not to a masked object — simpler: one warp, no cut-out, compositing or hole-filling
- build_affine unchanged (the centre is an input); spec and docstring updated
- Consequence: GeoBench's target masks are object-level, so they can't be ground truth for rotation/scale here; GeoBench stays as the external instruction → parameter test and as a source of sample images and masks
- Mask IoU becomes transform-consistency IoU: warp the same mask with the predicted and the true matrix and compare
- Added "the image" and "the picture" to the object words
- Object-level editing kept as a possible later extension


## 2026-10-14 · Splits
- build_splits.py: train / val / test (random 80/10/10) plus test_unseen_templates ("can you ..." patterns never seen in training)
- Sizes: train 4,376 · val 547 · test 548 · test_unseen_templates 1,464; 65 exact duplicates removed from 7,000
- Decision: validation split added — model settings are chosen on val, never on test
- Decision: unseen adverbs and verbs are tested on GeoBench, whose real instructions use words absent from our training vocabulary; replaced the planned word-swap test sets, which added complexity for a weaker test
- Overlap: 108 of 548 test sentences (20%) also appear in train with different numbers — intended; the unseen-template set and GeoBench measure unseen phrasing


## 2026-10-14 · GeoBench test file
- geobench.py: converts all 5,677 GeoBench 2D rows into our format in data/splits/geobench.jsonl — tx, ty / 512; θ = −rz; log sx
- Each row records its level (GeoBench adverb → small / medium / large) and whether the adverb is in our training vocabulary (adverb_seen), for the seen-vs-unseen-words comparison
- Instructions lower-cased to match training data; geobench_index kept so the image demo can find each row's picture


## 2026-10-16 · Text representations
- Each split is encoded once and saved to data/features/<split>_<rep>.npy, so training never re-runs CLIP or MiniLM.
- hand: 30 keyword flags (verbs, directions, adverbs) + 3 explicit numbers (pixels/512, degrees, log of factor) = 33 columns.
- tfidf: word and two-word counts, fitted on train only so val/test words don't leak into the vocabulary.
- clip: frozen CLIP ViT-B/32 text encoder, 512 columns, L2-normalised.
- minilm: frozen all-MiniLM-L6-v2, 384 columns.
- TF-IDF has 1,071 columns: it treats every number as a separate word ("by 155" and "by 156" are unrelated to it), unlike the hand features.
- Checked: "nudge the box slightly to the left, then rotate it by 30 degrees clockwise" → flags nudge, rotate, to the left, clockwise, slightly; numbers [0, 30, 0].


## 2026-10-16 · Baselines
- metrics.py: MAE of tx, ty (pixels at 512 width), θ (degrees, wrapped), log s, and mean corner displacement of a 512×512 image (one combined number, used for model selection).
- Constant baseline: train median of each target column, ignores the sentence.
- Ridge (StandardScaler + Ridge) per representation; alpha from {0.1, 1, 10, 100, 1000} chosen on val corner error. Results saved to outputs/baselines.csv.
- Val corner error: constant 170 px; ridge hand 83, clip 88, tfidf 90, minilm 93. Ridge halves the error but stays high: a linear model adds weights and cannot multiply direction by magnitude ("slightly" must be −0.09 for left, +0.09 for right). This is why the MLP is needed.
- test_unseen_templates has only single edits, so constant is lower there (142 vs 172 px). Compared as ridge ÷ constant: hand 0.49 → 0.49, tfidf 0.52 → 0.51, clip 0.54 → 0.71, minilm 0.54 → 0.61. Keyword features ignore the new phrasing; sentence embeddings are thrown off by it.
- GeoBench (unseen adverbs): constant 110 px; only hand beats it (92). tfidf 115, clip 190, minilm 322: unseen words push embeddings to regions the linear weights extrapolate badly from.
- Corner error is dominated by scale: corners sit ~362 px from the centre, so a 16% scale error moves them ~58 px. Always report the four separate MAEs too.


## 2026-10-16 · Gated MLP
- mlp.py: shared trunk Linear(n→256)–ReLU–Dropout(0.1)–Linear(256→256)–ReLU; presence head (3 logits) and value head (tx, ty, cos, sin, log s).
- Loss = BCE on presence + L1 on values masked to the edits that are present. L1 because the metric is MAE (pushes towards the median).
- Prediction is gated: an edit whose logit ≤ 0 is set to its do-nothing value.
- Inputs standardised with train mean/std. Adam lr 1e-3, batch 64, ReduceLROnPlateau (×0.5, patience 5), early stopping on val loss (patience 15, best weights restored), seed 0. Models saved to outputs/mlp_<rep>.pt, results to outputs/mlp.csv.
- Stopped at epoch: hand 91, tfidf 47, clip 126, minilm 91.
- Val corner error (constant 170): hand 83 → 42, clip 88 → 50, minilm 93 → 54, tfidf 90 → 65 (ridge → MLP). The non-linear model fixes the direction × magnitude problem.
- Presence F1 = 1.000 on synthetic val/test (hand, clip, minilm): detecting which edit is easy; the remaining error is magnitude.
- Phrasing sensitivity of embeddings mostly gone: clip MLP ÷ constant 0.32 on test, 0.28 on unseen templates (ridge: 0.54 → 0.71).
- GeoBench (constant 110): hand 92, minilm 97, tfidf 104, clip 115. Presence F1 drops to 0.74–0.82, so some GeoBench verbs are unseen.
- Hand features use the generator's own vocabulary, so their synthetic advantage is partly by construction; human-written instructions are the real test.
- Single seed: differences under ~5 px are not meaningful.


## 2026-10-16 · Breakdown and noise floor
- breakdown.py: MLP errors per group (category; level for single edits; on GeoBench also adverb_seen and verb_seen), saved to outputs/breakdown.csv.
- Oracle = perfect reader: copies explicit amounts, uses true direction + midpoint of the level's range for vague ones. Its error ≈ the lowest a text-only model can reach on synthetic data.
- Test corner: hand MLP 43.6 vs oracle 27.0. Single edits are near the floor (rotate 16.0 vs 15.2, translate 18.9 vs 15.9; unseen templates 23.0 vs 20.6).
- Combined edits hold most of the gap (hand 87 vs oracle 42): keyword flags have no word order, so the model cannot tell which adverb belongs to which verb (binding problem). Embeddings also fail (108–112). Candidate for novelty: split into clauses, predict each, compose.
- Explicit amounts: hand 5.6, clip 25.6, minilm 24.4, tfidf 77.7 (oracle 0). Embeddings read numbers only roughly; TF-IDF treats each number as a separate word.
- GeoBench: 2,661 / 5,677 rows (47%) start with an unseen verb (contract, swivel, drag, amplify, zoom, slide, expand). Presence accuracy on unseen verbs: hand 0.088, minilm 0.487, clip 0.470, tfidf 0.221 (seen verbs: hand 1.000, minilm 0.899).
- Hand features are best on known vocabulary; MiniLM generalises best to new verbs (translate presence 0.965). The demo will offer both.
- CLIP presence is low on GeoBench even for seen verbs (0.632; rotate 0.334). Hypothesis (not tested): new object nouns dominate the CLIP sentence vector.
- GeoBench verbs are not added to training: that would leak test vocabulary.
- To check: GeoBench large-level errors (~150–170 px) by category.


## 2026-10-16 · Image warping and IoU (D5)
- warp.py: applies [tx, ty, cos, sin, log s] to the whole image about its centre (bilinear for images, nearest for masks); iou() for binary masks.
- GeoBench ori_img is 512×512, so the px/512 translation conversion was correct. ori_mask is 640×640 and is resized to 512 (nearest) before use.
- showcase.py: transform-consistency IoU on all 5,677 GeoBench rows (mask warped by true vs predicted edit), saved to outputs/iou_geobench.csv; qualitative grid (2 per edit type, seed 0) saved to outputs/qualitative_geobench.png.
- Do-nothing reference (unmoved mask vs true result) added: IoU alone depends on object size and position, so it needs a reference.
- Mean IoU: hand 0.52, minilm 0.48, tfidf 0.47, clip 0.44, do nothing 0.46. Rotate: hand 0.69 vs 0.61; scale: hand 0.53 vs 0.44; translate: all ≈ do nothing (hand 0.32 vs 0.33).
- Translation fails because GeoBench directions ("leftward", "upward", "upper-left") never appear in training ("to the left", "up", "toward the top-right"): the edit is detected but moved the wrong way, which overlaps less than not moving.
- GeoBench tests three kinds of unseen words: verbs (presence fails), directions (wrong direction), adverbs (wrong size). Not added to training (test vocabulary leak). Novelty candidates: clause splitting for combined edits; paraphrase/synonym augmentation evaluated on human instructions.
- Mask alignment checked visually on the grid: outline sits on the object. Outlines are markers only (red = true edit, green = prediction); the whole image is warped, as the black borders show.


## 2026-10-16 · Generator v3 (after demo testing)
- Demo test "rotate the image clock-wise by 90 degrees" failed: no training rotation exceeded 39° (explicit range was 5–39), and a regressor does not extrapolate beyond its training range.
- Explicit ranges widened: rotate 5–180° (covers every distinct rotation), translate 20–300 px, scale ×1.1–4.0. Vague ranges unchanged (still calibrated to GeoBench): an explicit number carries its own size.
- More number formats: "px", "°", "deg", "2x", alongside "pixels", "degrees", "by a factor of".
- Neutral scale verbs "scale"/"resize", only with explicit factors: the factor gives the direction (2x bigger, 0.5x smaller). "Scale it slightly" has no direction, so it is never generated.
- Explicit scale in three forms: bigger (×1.1–4.0), smaller as the result ("to 0.7x", 0.25–0.91, 2 decimals), smaller as a divisor ("shrink by a factor of 2", shrink verbs only). The target uses the rounded number the sentence shows.
- hand_features: regexes read decimals ("22.5 degrees") and the new formats; neutral-verb flags added: 38 columns (35 flags + 3 numbers).
- Not added: object nouns. Objects will come from detection later; named objects are a stated limitation.
- Checked 20 generated samples by hand and hand features on 6 test sentences: all correct.


## 2026-10-16 · Gradio demo v1
- app.py: instruction + optional image (default: generated test card with grid, arrow and "TOP") + choice of hand or MiniLM. Shows the edited image, presence probabilities, tx/ty in px, θ, scale and the 2×3 matrix.
- Text is lowercased and stripped of end punctuation, as in training. Inputs standardised with the saved train mean/std. Only hand and MiniLM offered: TF-IDF's fitted vocabulary was not saved; CLIP was worst on GeoBench.
- Testing it found two problems, fixed in generator v3 and the next app version: rotations above 39° were never trained ("rotate 90 degrees" failed), and explicit pixel amounts were scaled by the uploaded image's width instead of meaning real pixels.


## 2026-10-17 · Pipeline v3: full re-run, 3 seeds, hand+MiniLM
- run_all.ps1 runs pytest → build_splits → geobench → encode → baselines → mlp → breakdown → showcase and stops at the first failure.
- v3 splits: train 4,424, val 553, test 553, unseen templates 1,468; 2 duplicates removed; TF-IDF now 2,233 columns (more distinct numbers).
- mlp.py trains every representation with seeds 0, 1, 2 and reports mean ± std (outputs/mlp_summary.csv); the seed-0 model is used by breakdown, showcase and the demo. Seed 0 reproduces the earlier single run exactly.
- New representation hand+minilm: the two saved feature files stacked side by side (35 + 384 = 419 inputs), same MLP. Added because each part has a measured weakness: hand is blind to unseen words, MiniLM is imprecise with numbers.
- Constant val corner 204 px (edits are larger in v3). Ridge val: hand 94, clip 111, minilm 113, tfidf 116.
- MLP corner (mean ± std), test / GeoBench: hand 44.8 ± 0.6 / 88.4 ± 1.1; hand+minilm 49.9 ± 1.1 / 77.9 ± 1.8; minilm 69.4 ± 2.1 / 110.3 ± 7.9; clip 70.2 ± 1.0 / 151.9 ± 16.2; tfidf 113.4 ± 2.4 / 108.1 ± 2.9. Oracle test 23.2.
- Hand is best on our vocabulary; hand+minilm is best on unseen vocabulary. Both gaps are several std wide.
- GeoBench presence on unseen verbs: hand 0.088, minilm 0.383, hand+minilm 0.679, better than either part. Overall presence 0.881 ± 0.035; IoU 0.555 (hand 0.536, do nothing 0.457).
- Explicit numbers (test, seed 0): hand 6.6, hand+minilm 28.9, minilm 50.1, clip 69.7. With 5–180° and 20–300 px, embeddings read numbers only roughly; in the concatenation the 3 exact-number columns are under-used (hypothesis).
- GeoBench translation: hand+minilm detects far more translations (0.974 vs 0.593) but unseen direction words make many go the wrong way: IoU 0.325 vs do nothing 0.333 (hand 0.360).
- Combined edits still the main gap (hand 97.5, hand+minilm 102.7, oracle 36.0). Single edits with hand near the floor (unseen templates 22.5 vs 19.6).
- Out-of-distribution instability: GeoBench std clip 16.2, minilm 7.9, hand 1.1. TF-IDF unstable even on val (std 6.7; stops at epoch 20–30).
- Demo default will be hand+minilm (failing slightly on numbers is less harmful live than doing nothing on an unseen verb); hand and minilm stay selectable.


## 2026-10-17 · Gradio demo v2
- Three text representations: hand+minilm (default), hand, minilm. The combined input is built exactly as in training (hand features then MiniLM, side by side).
- Explicit pixel amounts are converted to real pixels of the uploaded image: the model's tx, ty are fractions of a 512-px width, so they are multiplied by 512/W when the sentence states pixels. Vague amounts stay relative to image width.
- Spelling variants of direction words (clock-wise, anti clockwise, counterclockwise, …) are normalised to the training spelling before encoding. Demo input cleaning only; reported metrics use raw text.
- Output box shows the text actually used (after cleaning), presence probabilities, tx/ty in px, θ, scale, image size and the 2×3 matrix.
- `python app.py --share` gives a temporary public link for testing with others.
- Manual testing agrees with the measured results: minilm handles new translation verbs better, hand handles explicit scale factors better; the combination gets most of both.


## 2026-10-17 · Human-written test set
- 100 instructions written by people, labelled by the writers (direction, and amount as a number or small/medium/large); 98 kept, 2 out of scope (absolute position; size change in pixels). 149 labelled edits: 53 translate, 44 rotate, 52 scale.
- Label conventions: amounts with no stated size labelled medium (writers' rule); undefined units ("units", "cm", "inches") labelled none (presence and direction scored, amount not); percentages converted (e.g. reduce by 10% = x0.9, 50% of width = 256 px).
- eval_human.py scores hand, minilm and hand+minilm once, on raw lowercased text: presence, direction (translate within 22.5°, rotate by sign after wrapping, scale by sign of log s), explicit error (median), level hit for vague amounts. Per-edit predictions in outputs/human_edits.csv.
- All three presence flags right: hand 0.58, minilm 0.78, hand+minilm 0.78. Detection with MiniLM in the input 0.82–0.94 per edit type; hand alone 0.61–0.77.
- Direction is the main failure: translate 0.42–0.45, rotate 0.48–0.57, scale 0.62–0.81 (hand+minilm best on scale).
- Translation by labelled direction (hand): up 1.00, down 0.73, left 0.23, right 0.08. Training says "to the left/right"; people write bare "left/right", "rightwards". Diagonals have only 1–2 sentences each.
- Rotation: 26 of 44 human rotations say left/right/other instead of a clockwise word. Direction accuracy with clockwise words 0.83–0.89, without 0.19–0.38. The failure is vocabulary coverage, not model capacity.
- Explicit scale median error |log s| 0.30–0.41: people use %, "times", twice/double/half, which are not in training. Explicit rotation median error 41–47°, mostly from wrong direction, plus numbers without "degrees", 270° and radians.
- The set was scored once and is not used to change the model. Novelty plan: add left/right rotation, bare and "-wards" directions, %, times, twice/half to the generator, and evaluate on a fresh human set from different writers.