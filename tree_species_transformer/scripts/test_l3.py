import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from models.transformer import TreeTransformer
from dataset.tree_dataset import TreeSpeciesDataset

# ==== 配置 ====
DATA_PATH = "data"
MODEL_PATH = "checkpoints/best_model.pth"
INPUT_DIM = 1440
NUM_CLASSES = 19

# ==== 加载数据 ====
X = np.load(os.path.join(DATA_PATH, "X_1440.npy"))
y = np.load(os.path.join(DATA_PATH, "y_l3_1440.npy"))

val_ratio = 0.2
num_val = int(len(X) * val_ratio)
X_val = X[-num_val:]
y_val = y[-num_val:]

val_set = TreeSpeciesDataset(X_val, y_val)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

# ==== 加载模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TreeTransformer(
    input_dim=INPUT_DIM,
    model_dim=384,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==== 推理 ====
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# ==== 保存预测结果 ====
np.save("y_l3_preds.npy", y_pred)
np.save("y_l3_true.npy", y_true)
print("✅ Saved predictions and labels as .npy files")

# ==== 分类报告 ====
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report_l3.csv")
print("✅ Saved classification report to classification_report_l3.csv")

# ==== 混淆矩阵图 ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (L3)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
# plt.savefig("plots/confusion_matrix_l3.png", dpi=300)
print("✅ Saved confusion matrix to plots/confusion_matrix_l3.png")

# ==== 类别名称输出 ====
label_encoder = joblib.load(os.path.join(DATA_PATH, "label_encoder_l3_1440.pkl"))
class_names = label_encoder.classes_
print("🌲 Class labels (L3):", list(class_names))
