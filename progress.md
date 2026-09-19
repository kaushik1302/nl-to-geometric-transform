## 2026-09-20
- Env: Python 3.11.9, torch 2.14+cpu, CLIP (1,512), MiniLM (1,384). Smoke test OK.
- Decision: CPU-only torch — no GPU needed for prep-stage experiments.
- Decision: dataset generated not collected → data/ gitignored, seed + generator are the artifact.