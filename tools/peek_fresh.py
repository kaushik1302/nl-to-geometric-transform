import pandas as pd

for name in ("part_a_whole_image_edits", "part_b_object_edits"):
    df = pd.read_csv(f"data/human_fresh/{name}.csv", keep_default_na=False, dtype=str)
    print(name, len(df), "rows")
    for col in ("move_direction", "rotate_direction", "resize_direction"):
        print(" ", col, sorted(df[col].unique()))
    for col in ("move_amount", "rotate_amount", "resize_amount"):
        values = sorted(df[col].unique())
        print(" ", col, values[:25], "..." if len(values) > 25 else "")