import os
import glob
import json
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter

# ==== 配置 ====
DATA_DIR = "/home/group5/data-science-eo-regression/dataset"
SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)

pattern = os.path.join(DATA_DIR, "spectral_vegetation_dem_patch_3x3_growing_season_*.geojson")
files = glob.glob(pattern)
print(f"📂 Found {len(files)} geojson files.")

VALID_PREFIXES = (
    "B", "NDVI", "EVI", "SAVI", "MTCI", "MSR", "CIG",
    "RENDVI", "DEM", "SLOPE"
)

X_all, y_l1_all, y_l2_all, y_l3_all = [], [], [], []
expected_len = 1440  # ✅ 只保留 feature 向量长度为 1440 的样本

for fpath in tqdm(files, desc="Processing"):
    gdf = gpd.read_file(fpath)
    for _, row in gdf.iterrows():
        props = row
        feature_vector = []

        for k, v in props.items():
            if isinstance(k, str) and k.startswith(VALID_PREFIXES):
                try:
                    if isinstance(v, str) and v.strip().startswith("["):
                        v = json.loads(v)
                    flat = np.array(v).flatten()
                    feature_vector.extend(flat)
                except Exception:
                    continue

        if len(feature_vector) != expected_len:
            continue  # ❌ 丢弃非完整样本

        X_all.append(feature_vector)
        y_l1_all.append(props.get("l1_leaf_types"))
        y_l2_all.append(props.get("l2_genus"))
        y_l3_all.append(props.get("l3_species"))

# ==== 转数组 ====
X = np.array(X_all, dtype=np.float32)
print(f"✅ Final X shape: {X.shape}")

# ==== 标签编码 ====
enc_l1 = LabelEncoder()
enc_l2 = LabelEncoder()
enc_l3 = LabelEncoder()

y_l1 = enc_l1.fit_transform(y_l1_all)
y_l2 = enc_l2.fit_transform(y_l2_all)
y_l3 = enc_l3.fit_transform(y_l3_all)

# ==== 保存 ====
np.save(os.path.join(SAVE_DIR, "X_1440.npy"), X)
np.save(os.path.join(SAVE_DIR, "y_l1_1440.npy"), y_l1)
np.save(os.path.join(SAVE_DIR, "y_l2_1440.npy"), y_l2)
np.save(os.path.join(SAVE_DIR, "y_l3_1440.npy"), y_l3)

joblib.dump(enc_l1, os.path.join(SAVE_DIR, "label_encoder_l1_1440.pkl"))
joblib.dump(enc_l2, os.path.join(SAVE_DIR, "label_encoder_l2_1440.pkl"))
joblib.dump(enc_l3, os.path.join(SAVE_DIR, "label_encoder_l3_1440.pkl"))

# ==== 树种样本分布打印 ====
print("\n📊 树种（L3）样本分布:")
counter = Counter(y_l3)
classes = enc_l3.inverse_transform(sorted(counter.keys()))
for cls, count in zip(classes, [counter[k] for k in sorted(counter.keys())]):
    print(f"  • {cls:25s}: {count}")

print("\n🎉 Filtered 1440-dim data preparation complete!")
