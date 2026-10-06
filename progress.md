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