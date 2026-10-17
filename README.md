# Predicting Geometric Edit Parameters from Natural Language

Type an edit instruction such as *"rotate it 90 degrees clockwise and shrink it to 0.7x"*;
the model predicts the **translation, rotation and scale** it describes and applies them to an image.

IE 643 Deep Learning, IIT Bombay · Team **DyoOptix**: Kaushik Kuppili (26M1507), Akash Gudi (26M1516)

---

## How it works

1. **Text → numbers.** Each instruction is encoded with one of five text representations:
   hand-crafted keyword features, TF-IDF, frozen CLIP (ViT-B/32 text encoder), frozen MiniLM
   (all-MiniLM-L6-v2), or hand features + MiniLM side by side.
2. **Numbers → edit.** A small **gated multi-head MLP** predicts which edits are present
   (translate / rotate / scale) and their values `[tx, ty, cos θ, sin θ, log s]`.
   Edits judged absent are set to "do nothing".
3. **Edit → image.** The values become a 2×3 affine matrix applied to the **whole image about
   its centre** (OpenCV, bilinear for images, nearest for masks).

Training data is synthetic (7,000 generated instructions with exact targets). Evaluation uses
held-out synthetic splits, unseen sentence templates, the GeoBench 2D benchmark (unseen wording,
real images) and a set of 98 human-written instructions.

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
```

The trained models are included in `outputs/`, so the demo runs without training.

## Run the demo

```powershell
python app.py
```

Open the printed local URL (usually http://127.0.0.1:7860). Type an instruction, optionally upload
an image (otherwise a test card with a grid and an upward arrow is used), and choose a text
representation:

- **hand+minilm** (default): best on wording the model was not trained on
- **hand**: most precise with known words and exact numbers
- **minilm**: sentence embedding only

The output shows the edited image, the detected edits with their probabilities, tx/ty in pixels,
the angle, the scale factor and the affine matrix.

## Reproduce all results

```powershell
python tools/download_geobench.py   # once: GeoBench 2D into data/hf_cache
.\run_all.ps1
```

`run_all.ps1` runs, in order and stopping at the first failure: unit tests → data generation and
splits → GeoBench test file → text encoding → constant and ridge baselines → gated MLP (3 seeds)
→ per-group breakdown and noise-floor oracle → GeoBench image warping and IoU → human test set →
report figures. It takes roughly 15–25 minutes on a CPU. Results are written to `outputs/`
(CSV tables, models, `outputs/figures/`).

## Results

Corner displacement error of a 512×512 image (px, lower is better), gated MLP, mean ± std over
3 seeds:

| Representation | Test | Unseen templates | GeoBench |
|---|---|---|---|
| constant baseline | 214.5 | 176.1 | 109.6 |
| hand | **44.8 ± 0.6** | **22.5 ± 0.0** | 88.4 ± 1.1 |
| hand + MiniLM | 49.9 ± 1.1 | 30.8 ± 1.0 | **77.9 ± 1.8** |
| MiniLM | 69.4 ± 2.1 | 41.8 ± 0.9 | 110.3 ± 7.9 |
| CLIP | 70.2 ± 1.0 | 57.6 ± 2.4 | 151.9 ± 16.2 |
| TF-IDF | 113.4 ± 2.4 | 73.9 ± 3.6 | 108.1 ± 2.9 |
| oracle (noise floor) | 23.2 | 19.6 | – |

- Hand features are best on the training vocabulary; hand + MiniLM is best on unseen wording.
- On GeoBench images, transform-consistency IoU is 0.555 for hand + MiniLM and 0.536 for hand,
  against 0.457 for doing nothing.
- On the 98 human-written instructions, hand + MiniLM detects the right set of edits in 78% of
  sentences. Direction is the main weakness: 0.43 for translation, 0.57 for rotation, 0.81 for scale.

## Known limitations

- **Direction wording.** Rotations phrased as "turn left/right" and bare "left/right" for
  translation were not in the training vocabulary, so their direction is often wrong.
- **Number formats.** Percentages ("by 30%"), "times", "twice/half" and numbers without units
  ("rotate 45") are not reliably read.
- **Combined edits.** With two or three edits in one sentence, amounts can be attached to the
  wrong edit.
- **Scope.** Edits are relative and apply to the whole image. Absolute positions ("move it to the
  corner"), flips, and edits to one specific object are not supported.
- Ranges: explicit rotations up to 180°, translations up to 300 px (at 512 px width),
  scale ×0.25 to ×4.

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | Gradio demo |
| `run_all.ps1` | one-command pipeline |
| `src/dyooptix/data/` | vocabulary, magnitude ranges, data generator, splits, GeoBench conversion |
| `src/dyooptix/geom/` | affine matrix construction and image warping |
| `src/dyooptix/models/` | text encoding, metrics, baselines, gated MLP, evaluation, figures |
| `tests/` | unit tests for the affine maths |
| `tools/` | GeoBench download and inspection scripts |
| `spec/transform_spec.md` | coordinate and parameter conventions |
| `data/human/` | the human-written test set and its out-of-scope sentences |
| `outputs/` | trained models (generated tables and figures are not tracked) |
| `progress.md` | dated log of decisions and results |

## Conventions

Image coordinates with **+x right and +y down**; positive angles are **counter-clockwise** on
screen; translation is a fraction of image width (explicit pixel amounts refer to a 512-px-wide
image); scale is uniform. Full details in `spec/transform_spec.md`.