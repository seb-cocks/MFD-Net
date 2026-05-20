import pandas as pd
import numpy as np

file_path = "C:\\Apps\\Code\\web_crawler\\x-band-radar-unique.csv"

df = pd.read_csv(file_path)

# Get min and max for each numeric column
min_values = df.min(numeric_only=True)
max_values = df.max(numeric_only=True)

# Select only the last 3 columns
cols_to_scale = df.columns[-3:]

# Offsets you want to apply (5% and 10% in this example)
offsets = [
    # 0.025,
    0.05,
    0.1,
]

# Generate augmented datasets
augmented_dfs = []
for offset in offsets:
    df_lower = df.copy()
    df_lower[cols_to_scale] *= 1 - offset

    df_upper = df.copy()
    df_upper[cols_to_scale] *= 1 + offset

    augmented_dfs.extend([df_lower, df_upper])

# Concatenate all augmented versions
df_augmented = pd.concat(augmented_dfs, ignore_index=True)

# Count unique combos in the last 3 columns
unique_count = df_augmented[cols_to_scale].drop_duplicates().shape[0]
total_count = df_augmented.shape[0]

print(f"Total rows after augmentation: {total_count}")
print(f"Unique combos (last 3 cols): {unique_count}")
print(f"Duplicates (same combo appears more than once): {total_count - unique_count}")

output_file_path = "C:\\Apps\\Code\\web_crawler\\x-band-radar-augmented.csv"
df_augmented.to_csv(output_file_path, index=False)
