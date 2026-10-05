## 2026-09-20
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