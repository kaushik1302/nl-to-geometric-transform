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
- hand_features: regexes read decimals ("22.5 degrees") and the new formats; neutral-verb flags added: 35 columns (32 flags + 3 numbers).
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


## 2026-10-18 · Stage 1 code freeze
- make_figures.py draws four report figures from the saved results: representations (mean ± std, constant, oracle), ridge vs MLP, error by level vs oracle, human-set direction accuracy by phrasing. Added to run_all.ps1.
- Trained seed-0 models committed (outputs/mlp_*.pt, 0.3–2.6 MB) so the demo runs from a fresh clone without training.
- README: setup (CPU PyTorch index), demo, one-command reproduction, results table, limitations, layout, conventions.
- Fresh-clone test passed: new folder, new venv, install from requirements, pip install -e ., app starts with the committed models.
- Code frozen for Stage 1 and tagged stage1.


## 2026-11-18 · Generator v4: vocabulary and format coverage
- vocab.py: translate directions grouped by vector (TRANSLATE_PHRASES) so all 8 directions stay equally likely; added bare left/right, leftwards/rightwards/upwards/downwards. Rotation: left = counter-clockwise, right = clockwise; added counterclockwise, anti-clockwise. New formats: % (translate = fraction of width; scale = ×(1 ± n/100) by verb), "times" for enlarging, factor clauses (double / twice / half / halve).
- generate.py: a third of explicit translations written as %; two new explicit scale forms (percent, factor clause); factor clauses keep template_id −1. GeoBench words still not used.
- encode.py: keyword list de-duplicated (left/right serve translate and rotate); factor-word flags; "times" read as a factor; new percent column. Hand features 35 → 48 columns (44 flags + 4 numbers).
- tests/test_generate.py: left/right signs for rotate and translate; % translate equals n/100 of width. 10 tests pass.
- v4 splits: train 4,470, val 559, test 559, unseen templates 1,369; 43 duplicates removed (mostly repeated factor clauses); 88 test sentences also in train.
- GeoBench corner error (v3 → v4): hand 88.4 → 81.4, hand+minilm 77.9 → 78.0, minilm 110.3 → 105.8, tfidf 108.1 → 92.6, clip 151.9 → 139.1.
- Old human set (development only: v4 was designed from its failures): all three presence flags right, hand+minilm 0.78 → 0.90; direction translate → 0.72, rotate → 0.84, scale → 0.83; explicit rotation median error 41–47° → 15° (hand). To be confirmed on the fresh human set.
- Synthetic test errors are not comparable with v3 (different generator; oracle floor 23.2 → 25.4).
- Explicit translate error on human set still large (median 45–61 px, n = 15): target for number fusion.


## 2026-11-19 · Clause splitting (inference only)
- split.py: split_clauses cuts at and / then / and then / commas / semicolons; "up and right", "up and to the left" are protected as one diagonal direction. merge adds tx, ty, angles and log s over the clauses: exact, because rotation and scale act about the object's own centre (t = t1 + t2, θ = θ1 + θ2, s = s1·s2, any order). No retraining: the v4 models are used as they are.
- tests/test_split.py: joiners, diagonal protection, merge of moves/angles/log scales, one clause unchanged. 14 tests pass.
- eval_split.py: whole vs split with the same models and features; added to run_all.ps1 (and the missing comma after the showcase step fixed).
- Synthetic test, combined sentences, corner error whole → split: hand 106.7 → 45.4 (oracle floor 40.4), hand+minilm 103.8 → 55.4, minilm 155.4 → 72.7. Single-edit sentences, unseen templates and GeoBench identical in both modes.
- Old human set, multi-edit sentences (development only), hand+minilm whole → split: all presence flags right 0.775 → 0.975; direction rotate 0.833 → 0.958, scale 0.788 → 0.970, translate 0.588 → 0.765. Explicit median error: rotate 25.9° → 11.2°, scale 0.197 → 0.156.
- Known limits: "rotate and shrink it" gives a first clause with no direction or amount; explicit translate error still 35–37 px (target for number fusion).
- Demo: hand+minilm chosen automatically (best or tied on presence, direction and GeoBench); representation selector and split checkbox moved to a collapsed "Advanced (for comparison)" panel; clauses shown in the output; three-clause example added.


## 2026-11-19 · Number-preserving fusion (inference only)
- New `models/numbers.py`: `fix_numbers(clause, pred, present)` replaces only the *magnitude* of a clause's prediction with the exact number in the text (deg/rad, px, %, x/times/factor, half/double/twice); the network keeps edit type and sign. Runs per clause before the clauses are added up.
- Rules: "to n%" = n/100, "by n%" = 1±n/100; any angle wrapped into (−180°, 180°]; rotation with no direction word defaults to counter-clockwise; a unit-less number is px for moves and degrees for rotation; diagonal moves snap to the nearest of 8 directions and the number is the total length; a "%" goes to scale if the clause has a scale word, otherwise to translation.
- No retraining. eval_split now compares whole / split / split+numbers; edit rows record sentence and amount.
- Results, hand+minilm, synthetic corner error (px): test combined 103.8 → 55.4 → 43.3; test single 30.0 → 30.0 → 20.4; unseen templates 33.2 → 33.2 → 24.4; GeoBench unchanged at 76.4 (no numbers or combined edits there).
- Human set (development only), hand+minilm, explicit amounts: translate mean error 0.0 px (15 edits), scale 0.000 (35), rotate exact except two sentences. Direction accuracy: rotate 0.85 → 0.95 (single), 0.96 → 1.00 (multi); scale 0.895 → 0.947 and 0.97 → 1.00.
- Known failure: word-only angles ("turn it upside down" = 180°) are outside the training vocabulary. Limitations: "move it to 100 px" is read as a relative move; "half" as ×0.5 even for "enlarge by half".

## 2026-11-19 · Object clamp (4a)
- New `geom/clamp.py`: keeps an edited object fully inside the image. Order: scale (grows about the object's own centre until the first side touches the border; scale-down never clamped; limit measured at the object's current position), then rotation (reduced toward 0° only if the object is too large to turn), then translation (clipped to the range that keeps the turned, scaled outline inside). Each change gives a ⚠ warning text. Whole-image edits are never clamped.
- `place()` reuses `build_affine`, so the clamp and the image warp always agree. 5 tests in `tests/test_clamp.py` (31 tests in total).
- Smoke scripts moved to `tools/` (smoke_gsam.py, smoke_lama.py); cache folder from DYOOPTIX_CACHE, default data/hf_cache.

## 2026-11-19 · Text-grounded object masks (4b)
- New `geom/grounding.py`: `find_object(image, word)` runs Grounding DINO (word → box, on a shrunken copy of the photo) then SAM (box → mask); `outline(mask)` gives the convex hull of the largest blob for the clamp. Models load once (`load()`), results are cached per (photo fingerprint, word), and a missing object raises `ObjectNotFound` with the best rejected score ("best match 0.21, needs 0.30").
- LaMa tests on a large cup photo: inpainting at full size took 42 s with a grey band and seam; the same fill on a crop shrunk to 512 px took 5 s and removed the tint. Decision: object crop plus margin, long side at most 512, never enlarged; only masked pixels are written back. Known limit: printed text and fine patterns (keyboard mat) stay visible in the fill.
- `tools/smoke_gsam.py` now checks the module (load time, cache, outline, missing object). 33 tests pass.


## 2026-11-19 · Hole filling with LaMa (4c-i)
- New `geom/inpaint.py`: `fill(image, mask)` grows the mask by about 7 px, crops the hole plus a 50% margin, shrinks the crop to at most 512 px (never enlarges), fills it with LaMa, scales the fill back and writes only the hole's pixels into a copy of the photo. LaMa loads once (`load()`).
- Cup photo (1536 × 1536, hole 24% of the photo): fill 7.6 s on the first call after loading, result identical to the 512-px smoke test.
- `tools/smoke_lama.py` now checks the module.


## 2026-11-19 · Object editing pipeline (4c-ii)
- New `geom/object_edit.py`: `edit_object(image, mask, target)` takes the model's [tx, ty, cos, sin, log s], uses the mask centroid as the pivot, clamps the edit with `clamp()`, fills the old place with LaMa (`fill()`), moves the object with the same `build_affine` matrix and pastes it with a soft edge (mask eroded 1 px, blurred with sigma 1.5 px). No edit left after clamping returns the photo untouched. 4 tests with a stand-in for LaMa (37 tests in total).
- Cup photo (1536 × 1536), asked move +800 px, rotate +20°, scale ×1.3: applied +469 px (right edge), +20°, ×1.16 (left edge), both with ⚠ warnings; edit 14.9 s after the models are loaded.
- Known limits: 2D edit only, so a turned or moved object can look like it floats (no contact, shadow or depth); the scale limit is measured before the move, so a move away from the edge does not allow more growth.
- `tools/smoke_edit.py`: runs grounding + editing on a photo with an edit typed as numbers.


## 2026-11-20 · Scene-derived translation: touch and beside (4e-i)
- Novelty feedback from the intensive assessment: derive a geometric parameter from the scene when the instruction gives no number ("move the circle right until it touches the square").
- New `geom/contact.py`: `touch(mover, target, d)` slides the mover along d until it touches (no gap); with no direction it heads for the target's centre; refuses when the target is behind or out of the path. `beside(mover, target, side)` aligns the bottoms and places the mover on the nearest side (or the named side) with a gap of 10% of the narrower width. The first object moves, so "bottle beside the mug" and "mug beside the bottle" differ.
- Method: lanes 1 px wide at right angles to the motion; the smallest gap over lanes that hold both objects is the contact distance. 7 tests (44 in total); 0.08 s on two 1536-px masks.

## 2026-11-20 · Reading relations from the text (4e-ii)
- New `models/relations.py`: `parse(clause)` finds "until it touches/reaches/hits/meets/contacts the X" (touch) and "beside / next to / alongside / near / close to / adjacent to / by the side of the X" or "to the left/right of the X" (beside, with a named side). Returns the mover (named first, so it is the one that moves), the target, the side, whether a direction word is present, and the rest of the clause for the text model. Plain "to the left" without "of" stays an ordinary move; plain "by" is excluded because it means an amount.
- `name(text)`: one- or two-word object name after the/this/that/my; "it" and image/picture/photo mean the whole image.
- Rules written from our own examples, not tuned on the fresh test set. 6 tests (50 in total).

## 2026-11-20 · Relation to translation (4e-iii a)
- `contact.relation_shift(rel, mover, target, pred)`: beside → `beside()`; touch with a direction word → the text model's direction snapped to the nearest of 8 directions, then `touch()`; touch without one → toward the target's centre. Not possible (target behind or out of the path, no move detected) → ValueError with the reason. The model decides which way, the scene decides how far. 2 tests (52 in total).


## 2026-11-20 · Shape test set for scene-derived edits (4e-iii b1)
- New `data/shapes.py`: 48 images (512 px, red circle, blue square, fixed seed 643): 16 "move the circle right/left/up/down until it touches/reaches/hits the square" (4 meeting the square's corner), 8 without a direction word, 16 "beside / next to / near / alongside / close to" (nearest side), 8 "to the left/right of" (named side, sometimes the far side).
- Expected translation from exact circle–rectangle geometry, written independently of contact.py. Cross-check with true masks and the correct direction: contact.py agrees within 0.86 px (mean 0.09 px).
- Output in data/shapes (git-ignored; rebuild with `python -m dyooptix.data.shapes`).


## 2026-11-20 · Touching the photo's edge
- `relations.parse`: "until it touches/reaches/hits the edge / end / border / boundary / side", optionally with left / right / top (upper) / bottom (lower) → kind "edge".
- `contact.to_edge(mask, side, d)`: named side → that border; a direction word only → the first border reached along it; neither → the nearest border. A named side wins over a direction word.
- Shape test set extended to 56 images with 8 edge cases (4 named edges, 2 direction-only, 2 nearest); expected moves exact. 54 tests in total.

## 2026-11-20 · Scene pipeline and shape evaluation (4e-iii b2)
- New `models/scene.py`: `read(sentence)` splits clauses, finds the relation and the mover (first-named object; "it" refers to it); `scene_edit(...)` merges plain clauses (px = pixels of this photo), clamps rotation and scale first, measures the relation on the turned and scaled mask, and lets the relation decide the final position. 4 tests.
- New `models/eval_scene.py`: shape test set with true masks or Grounded-SAM masks; scores the clamped translation against the exact expected one (within 3 px).
- Found on the shape test: "slide the circle left" gave a weak direction (155.6°) that the 8-way snap turned into down-left. New `numbers.snap(tx, ty, text)`: diagonal only if the clause names one (a horizontal and a vertical word, or "diagonal"), else nearest of the 4 main directions; used in `fix_numbers` and `relation_shift`. Rule added after seeing this failure; the fresh LLM-written set is untouched.
- Results: shapes with true masks 56/56 within 3 px (was 54/56). Human set (development): translate direction hand+minilm multi-edit 0.765 → 0.794, minilm single-edit 0.526 → 0.684; explicit amounts still exact; nothing worse.


## 2026-11-20 · Two named objects must be two different things
- Shape test with Grounded-SAM masks exposed: for "square", DINO-tiny returned the circle's box (score 0.65–0.68 vs 0.91 for "circle") in 3 of 56 images, so mover and target were the same mask (two refusals, one wrong side).
- `grounding.find_object(image, name, avoid=box)`: a candidate box overlapping `avoid` with IoU ≥ 0.5 is skipped; the best remaining box is used, else "couldn't find X apart from the other object". The target is looked up avoiding the mover's box. `box_iou` + test. Rule added after seeing this failure; the fresh LLM-written set is untouched.
- Shape test (56 images, sentences with no number, expected moves from exact geometry), within 3 px of the exact translation: true masks 56/56 (median 0–0.24 px), Grounded-SAM masks 56/56 (median 1.0–1.4 px; was 53/56 before this rule).


## 2026-11-20 · Object names without "the"; "near to"
- BESIDE accepts "near to"; names written without "the" are found (first word that is not a verb, direction or number unit)
- names_whole() separates "image named" (whole image) from "nothing named" (main object, next in the demo step)

## 2026-11-20 · Main object when no object is named
- find_main: Grounding DINO with the word "object"; boxes of 1%-90% of the photo; the best box needs score >= 0.30, none means the whole image (message says so)
- Several similar objects (score within 0.10, or score >= 0.15 and area within half to twice the best, not wider than 90% of the photo) = tie: the biggest is edited, all are marked, message says so; one clear object: no message
- Values are untuned defaults fitted on the cup, shape images and three dev photos only; the fresh test set was not used
- Known limit: the detector knows only the word "object", so the message cannot name the object that was edited

## 2026-11-20 · One pipeline for the demo and the scorers
- pipeline.run(image, instruction, rep, use_split, object_name): whole-image edit or object edit (ground, text model, scene_edit, clamp, inpaint, paste); errors leave the photo unchanged with the reason
- Object box replaces the mover only; a relation's target still comes from the sentence

## 2026-11-20 · Demo wired to the pipeline
- app.py only builds the interface and calls pipeline.run; models, DINO+SAM and LaMa load once at start-up
- New Object box (replaces the mover only); "What the app considered" image shows boxes only when several similar objects were found
- Test card treated like any photo: whole-image examples say "the image"; square and circle examples edit one object
- Object report shows the affine matrix about the object's centre and the clamp warnings

## 2026-11-20 · Touch without a direction slides along the main axis
- Objects stay on their surface (no lift toward the target's centre); falls back to the straight line when no lane is shared
- Shape set 56/56 with true masks (median error 0.0 px for touch without direction)


## 2026-11-20 · Figure: one sentence, four gaps
- fig_scene.py runs one instruction ("move the circle right until it touches the square") on four images that differ only in the gap (40, 100, 180, 260 px) and plots the predicted move against the exact one
- Predicted +40 / +100 / +180 / +260 px against exact +39 / +99 / +179 / +259 px: the move comes from the picture, not from a typed number (1 px difference is the SAM mask offset)

## 2026-11-20 · Poster figure: demo examples with predicted parameters
- tools/fig_demo.py runs six instructions through pipeline.run and draws input, predicted parameters and edited result per row: two whole-image edits (37 degrees turn; shrink and turn on a non-square photo), a named object moved by pixels, rotate and enlarge with clamp warnings, touch, beside
- Only the test card and dev photos are used; the fresh test set is untouched


## 2026-11-21 · Vague moves snapped to a main direction; fresh-set scorer
- fix_numbers snaps every move to the 4 main directions (8 if the clause names a diagonal); a vague move keeps the model's length
- Found on a made-up dev sentence ("slide the book a little to the left" drifted 28 degrees down); fresh set not used
- hand+minilm: human-set move direction (multi-edit) 0.794 -> 0.912; corner error test 28.4 -> 28.0, unseen 24.4 -> 24.1, GeoBench unchanged
- pipeline.run returns the final edit (params) and the edited object's name for scoring
- eval_fresh.py scores Part A (whole image, 512 px) and Part B (object edits on the photos, clamped rows separate)
- Rules fixed before scoring: "to n%" = n/100, "by n%" = 1 +/- n/100 with the label's direction; an edit counts as detected when it is non-zero in the final output


## 2026-11-21 · Fresh test set scored once

- Set: 100 whole-image sentences (Part A, 512 blank image) + 60 object-level sentences on 10 photos (Part B). Written by an LLM, scored once, nothing tuned afterwards.
- Part A (hand+minilm): presence all right 0.970. Translate direction 0.683 (explicit median error 0 px). Rotate direction 1.000 (explicit 0°). Scale detected 0.94, direction 0.780.
- Part B: right object found 58/58 (the 2 absent-object rows are correctly refused). Unclamped translate direction 0.696, rotate direction 1.0, scale direction 0.889. Clamped rows (18) reported separately.
- Main failure causes: diagonal moves collapse onto one axis, "upward/upper" read as down, exact factor inverted when the model guesses shrink, no size word in the sentence, idioms or a missed scale clause, clamping changing the move.
- Caveats: LLM-written set is optimistic; rules (snap, avoid, main-object thresholds) were added after seeing dev failures.
- Files: data/human_fresh/, outputs/fresh_human_fresh_*.csv, src/dyooptix/models/eval_fresh.py.