import os
import pandas as pd
import shutil
from tqdm import tqdm

# -------- PATHS --------
metadata_file = "HAM10000_metadata.csv"

image_dirs = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]

output_dir = "dataset"

# -------- LOAD METADATA --------
df = pd.read_csv(metadata_file)

# -------- CREATE OUTPUT FOLDERS --------
classes = sorted(df['dx'].unique())
for c in classes:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

print("Classes found:", classes)

# -------- CONVERT DATASET --------
missing = 0
copied = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row['image_id']
    label = row['dx']
    filename = img_id + ".jpg"

    found = False

    for folder in image_dirs:
        src = os.path.join(folder, filename)

        if os.path.exists(src):
            dst = os.path.join(output_dir, label, filename)

            # avoid duplicate copy
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

            found = True
            break

    if not found:
        missing += 1

# -------- SUMMARY --------
print("\n✅ Conversion completed!")
print(f"Total copied: {copied}")
print(f"Missing images: {missing}")