import geopandas as gpd
import os
import glob
from collections import Counter

# 设置数据路径
DATA_DIR = "/home/group5/data-science-eo-regression/dataset"
pattern = os.path.join(DATA_DIR, "spectral_vegetation_dem_patch_3x3_growing_season_*.geojson")

# 匹配所有目标文件
geojson_files = glob.glob(pattern)
print(f"🔍 Found {len(geojson_files)} GeoJSON files")

# 统计信息
species_counter = Counter()
feature_dims = set()
uuid_missing = []

for filepath in geojson_files:
    filename = os.path.basename(filepath)
    print(f"\n📂 File: {filename}")

    gdf = gpd.read_file(filepath)
    
    # 打印条数
    print(f"  ➤ Number of samples: {len(gdf)}")
    
    # 检查 feature 列
    if 'features' in gdf.columns:
        feat_example = gdf.iloc[0]['features']
        feature_dims.add(len(feat_example))
        print(f"  ➤ Feature vector length: {len(feat_example)}")
    else:
        print("  ⚠️ No 'features' column found!")

    # 检查 uuid
    if 'uuid' not in gdf.columns:
        uuid_missing.append(filename)
        print("  ⚠️ No 'uuid' column!")

    # 统计 tree species
    if 'species' in gdf.columns:
        species = gdf['species'].value_counts().to_dict()
        print("  ➤ Species counts (in file):", species)
        species_counter.update(gdf['species'].tolist())
    else:
        print("  ⚠️ No 'species' column for per-sample labels")

# 总览
print("\n📊 Global Stats:")
print(f" - Unique feature dims: {feature_dims}")
print(f" - Files missing uuid: {uuid_missing}")
print(f" - Total species distribution:")
for sp, count in species_counter.most_common():
    print(f"   • {sp:25s}: {count}")
