import os
import glob
import json
import numpy as np
import geopandas as gpd
from collections import Counter, defaultdict
from tqdm import tqdm

DATA_DIR = "/home/group5/data-science-eo-regression/dataset"
pattern = os.path.join(DATA_DIR, "spectral_vegetation_dem_patch_3x3_growing_season_*.geojson")
files = glob.glob(pattern)

VALID_PREFIXES = (
    "B", "NDVI", "EVI", "SAVI", "MTCI", "MSR", "CIG", "RENDVI", "DEM", "SLOPE"
)

print(f"🔍 Found {len(files)} files.")
field_counter = Counter()
dim_counter = Counter()
missing_feature_samples = 0
missing_label_samples = 0

field_presence_matrix = defaultdict(int)

for fpath in tqdm(files, desc="Analyzing"):
    gdf = gpd.read_file(fpath)
    for _, row in gdf.iterrows():
        props = row
        feature_vector = []
        fields_this_sample = []

        for k, v in props.items():
            if isinstance(k, str) and k.startswith(VALID_PREFIXES):
                try:
                    if isinstance(v, str) and v.strip().startswith("["):
                        v = json.loads(v)
                    flat = np.array(v).flatten()
                    feature_vector.extend(flat)
                    fields_this_sample.append(k)
                except:
                    continue

        if len(feature_vector) == 0:
            missing_feature_samples += 1
            continue

        dim_counter[len(feature_vector)] += 1
        for f in fields_this_sample:
            field_counter[f] += 1
        for f in set(fields_this_sample):
            field_presence_matrix[f] += 1

        if (props.get("l1_leaf_types") is None or
            props.get("l2_genus") is None or
            props.get("l3_species") is None):
            missing_label_samples += 1

# === 统计输出 ===
print("\n📊 特征维度分布：")
for dim, count in sorted(dim_counter.items()):
    print(f"  • {dim} dims → {count} samples")

print(f"\n❌ 无有效特征的样本数量: {missing_feature_samples}")
print(f"❌ 缺少 L1/L2/L3 标签的样本数量: {missing_label_samples}")

print("\n📌 字段使用频率 TOP 15:")
for field, count in field_counter.most_common(15):
    print(f"  • {field:20s}: {count}")

print(f"\n总样本数（有特征者）: {sum(dim_counter.values())}")
print(f"共同字段总数: {len(field_counter)}")
