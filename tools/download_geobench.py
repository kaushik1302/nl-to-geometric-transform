from datasets import load_dataset

ds = load_dataset("CIawevy/GeoBench", "2d", cache_dir="data/hf_cache")
print(ds)