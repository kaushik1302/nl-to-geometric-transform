# Re-run the whole pipeline, from data to figures.
# Run from the project root with the venv active:  .\run_all.ps1
$steps = @(
    "pytest",                         # unit tests first: stop early if the maths is broken
    "dyooptix.data.build_splits",     # generate 7,000 examples and split them
    "dyooptix.data.geobench",         # GeoBench test file (reads the local Hugging Face cache)
    "dyooptix.models.encode",         # hand, TF-IDF, CLIP, MiniLM features for every split
    "dyooptix.models.baselines",      # constant + ridge
    "dyooptix.models.mlp",            # gated MLP, 3 seeds
    "dyooptix.models.breakdown",      # per-group errors + noise-floor oracle
    "dyooptix.models.showcase"        # GeoBench IoU + qualitative grid
    "dyooptix.models.showcase",       # GeoBench IoU + qualitative grid
    "dyooptix.models.eval_human",     # human-written test set
    "dyooptix.models.make_figures"    # report figures
)

foreach ($step in $steps) {
    Write-Host "`n=== $step ===" -ForegroundColor Cyan
    python -m $step
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed at $step" -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nAll steps finished." -ForegroundColor Green