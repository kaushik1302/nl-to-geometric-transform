# Predicting Geometric Edit Parameters from Natural Language

Type an edit instruction such as *"move the cup 100 px to the right and rotate it 20 degrees clockwise"*;
the system predicts the **translation, rotation and scale** it describes, finds the object it names in
the photo, and applies the edit. Some amounts come from the picture itself:
*"move the circle until it touches the square"* moves exactly as far as the gap in that image.

IE 643 Deep Learning, IIT Bombay · Team **DyoOptix**: Kaushik Kuppili (26M1507), Akash Gudi (26M1516)

---

## How it works

**Text lane (Stage 1, plus inference-time rules)**

1. **Clauses.** A combined instruction is split at *and / then / commas*; each clause is predicted
   separately and the edits are added up (exact, because rotation and scale act about the object's
   own centre). Inference only, no retraining.
2. **Text → numbers.** Each clause is encoded with hand-crafted keyword features (44 flags + 4 numbers)
   and frozen MiniLM (all-MiniLM-L6-v2) side by side. A small **gated multi-head MLP** predicts which
   edits are present (translate / rotate / scale) and their values `[tx, ty, cos θ, sin θ, log s]`.
   Edits judged absent are set to "do nothing".
3. **Exact numbers.** When a clause writes an amount (degrees, radians, px, %, ×, times, double/half),
   that number replaces the network's guess; the network keeps the edit type and direction.
   Moves are snapped to the 4 main directions (8 when the clause names a diagonal).

**Image lane (novelty)**

4. **Which object.** Grounding DINO turns the object word into a box and SAM turns the box into a mask
   (Grounded-SAM). If no object is named, the main object is edited; "the image / picture / photo"
   edits the whole image. A named object that is not in the photo is refused with a message.
5. **Edits from the scene.** "until it touches the X", "beside / next to / near the X",
   "to the left of the X" and "until it touches the left edge" are measured from the masks:
   the text gives the direction, the picture gives the distance.
6. **Clamp.** An object edit is reduced, with a ⚠ warning, so the object stays inside the photo
   (scale, then rotation, then translation).
7. **Edit → image.** One 2×3 affine matrix. Whole image: warped about its centre. One object: LaMa fills
   the hole it leaves and the moved object is pasted with a soft edge.

Training data is synthetic (7,000 generated instructions with exact targets). Evaluation uses held-out
synthetic splits, unseen sentence templates, the GeoBench 2D benchmark (unseen wording, real images),
98 human-written instructions, a shape test set with exact expected moves, and a fresh LLM-written test set.

## Setup

Python 3.11. Commands below are for Windows PowerShell; on Linux/macOS activate with
`source .venv/bin/activate`.

```powershell
git clone https://github.com/kaushik1302/nl-to-geometric-transform.git
cd nl-to-geometric-transform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
pytest                               # 74 tests; no model downloads
```

The trained text models are included in `outputs/`, so the demo runs without training.
Grounding DINO, SAM and LaMa download on first use (over 1 GB in total) into `data/hf_cache`;
set the environment variable `DYOOPTIX_CACHE` to put them elsewhere.

## Run the demo

```powershell
python app.py            # local: http://127.0.0.1:7860
python app.py --share    # also a temporary public link
```

Type an instruction, optionally upload a photo (otherwise a test card with a grid, a square, a circle
and an upward arrow is used), and optionally type an object in the **Object** box to choose which object
moves. The output shows the edited image, the detected edits with their probabilities, tx/ty in pixels,
the angle, the scale factor, the affine matrix and any clamp warnings. When several objects look equally
like the main one, a second image shows the candidates.

The best text model (hand features + MiniLM) is used automatically. **Advanced** lets you compare
hand / MiniLM alone and turn clause splitting off.

On a CPU a whole-image edit is instant; an object edit on a large photo takes about 15 s once the models
are loaded (the first click on a photo also runs detection and segmentation).

Conventions used by the demo: a rotation with no direction word is counter-clockwise; a number with no
unit is pixels for moves and degrees for turns; "to 75%" means ×0.75 and "by 30%" means ×1.3 or ×0.7;
"N px" means N real pixels of the uploaded photo.

## Reproduce the results

```powershell
python tools/download_geobench.py   # once: GeoBench 2D into data/hf_cache
.\run_all.ps1
```

`run_all.ps1` runs, in order and stopping at the first failure: unit tests → data generation and
splits → GeoBench test file → text encoding → constant and ridge baselines → gated MLP (3 seeds)
→ per-group breakdown and noise-floor oracle → GeoBench image warping and IoU → human test set →
whole vs split vs split + exact numbers → shape test set → scene-derived edits on the shape set →
report figures. Results are written to `outputs/`.

Run by hand (the fresh-set scorer overwrites the committed one-time results):

```powershell
python -m dyooptix.models.fig_scene      # one sentence, four gaps: outputs/fig_scene_gaps.png
python tools/fig_demo.py                 # demo examples with predicted parameters
python -m dyooptix.models.eval_fresh     # fresh LLM-written test set (scored once on 21 Nov 2026)
python tools/fig_fresh_charts.py
python tools/fig_fresh_tables.py
```

`git checkout stage1` gives the exact code submitted for the intensive assessment.

## Results

**Stage 1** (generator v3). Corner displacement error of a 512×512 image (px, lower is better),
gated MLP, mean ± std over 3 seeds:

| Representation | Test | Unseen templates | GeoBench |
|---|---|---|---|
| constant baseline | 214.5 | 176.1 | 109.6 |
| hand | **44.8 ± 0.6** | **22.5 ± 0.0** | 88.4 ± 1.1 |
| hand + MiniLM | 49.9 ± 1.1 | 30.8 ± 1.0 | **77.9 ± 1.8** |
| MiniLM | 69.4 ± 2.1 | 41.8 ± 0.9 | 110.3 ± 7.9 |
| CLIP | 70.2 ± 1.0 | 57.6 ± 2.4 | 151.9 ± 16.2 |
| TF-IDF | 113.4 ± 2.4 | 73.9 ± 3.6 | 108.1 ± 2.9 |
| oracle (noise floor) | 23.2 | 19.6 | – |

**Novelty** (generator v4 + inference rules, hand + MiniLM):

- Combined edits, synthetic test, corner error: 103.8 px (whole sentence) → 55.4 (clause splitting)
  → 43.3 (+ exact numbers); noise floor 40.4.
- Scene-derived edits ("until it touches", "beside", "until the edge") on 56 shape images with no number
  in the sentence: **56/56 within 3 px** of the exact geometric move, with true masks and with
  Grounded-SAM masks. The same sentence on four gaps moves +40 / +100 / +180 / +260 px
  (exact +39 / +99 / +179 / +259).
- Fresh LLM-written set, scored once. Part A (100 whole-image sentences): all edit flags right 97%;
  direction right translate 68%, rotate 100%, scale 78%; written amounts median error 0 px / 0°.
  Part B (60 object sentences, 10 photos): right object 58/58, both absent objects refused.
  The set was written by an LLM, so these numbers are likely optimistic.

## Known limitations

- **Diagonal moves** are often kept on one axis (19 of the 41 direction errors on the fresh set).
- **"upward", "leftward"** (GeoBench-style words without the "s") are not in the training vocabulary.
- **Vague sizes**: "slightly" from a new writer often lands in another size level.
- **Scope**: one object per instruction; 2D edits only (no shadows or depth); no absolute positions
  ("to the corner"), flips or word-only angles ("upside down").
- **Speed**: object edits take seconds on a CPU.
- Ranges learned by the network: rotations up to 180°, moves up to 300 px at 512 width, scale ×0.25 to ×4
  (written numbers are copied exactly, so they are not limited by these ranges).

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | Gradio demo |
| `run_all.ps1` | one-command pipeline |
| `src/dyooptix/data/` | vocabulary, magnitude ranges, data generator, splits, GeoBench conversion, shape test set |
| `src/dyooptix/geom/` | affine matrix, image warping, clamp, Grounded-SAM grounding, LaMa inpainting, object editing, contact geometry |
| `src/dyooptix/models/` | text encoding, metrics, baselines, gated MLP, evaluations, clause splitting, exact numbers, relations, scene, pipeline, figures |
| `tests/` | 74 unit tests |
| `tools/` | GeoBench download and statistics, figure scripts; `tools/smoke/` smoke scripts |
| `spec/transform_spec.md` | coordinate and parameter conventions |
| `data/human/` | the 98-sentence human-written test set and its out-of-scope sentences |
| `data/human_fresh/` | the fresh LLM-written test set (100 + 60 sentences, 10 photos) |
| `samples/` | development photos used by `tools/fig_demo.py` |
| `outputs/` | trained models, poster figures and the fresh-set scores |
| `progress.md` | dated log of decisions and results |

## Conventions

Image coordinates with **+x right and +y down**; positive angles are **counter-clockwise** on
screen; translation is a fraction of image width (explicit pixel amounts are real pixels of the image);
scale is uniform; object edits turn and resize about the object's centroid. Full details in
`spec/transform_spec.md`.
