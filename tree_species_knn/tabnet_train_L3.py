import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 设置随机种子
np.random.seed(42)

# === 加载数据 ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tree_species_transformer/data"))
X = np.load(os.path.join(base_path, "X_1440.npy"))
y = np.load(os.path.join(base_path, "y_l3_1440.npy"))
label_encoder = joblib.load(os.path.join(base_path, "label_encoder_l3_1440.pkl"))
class_names = label_encoder.classes_

# === 数据划分 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 初始化模型 ===
clf = TabNetClassifier(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":20, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
    seed=42
)

# === 训练模型 ===
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)

# === 推理 ===
y_pred = clf.predict(X_test)

# === 评估报告 ===
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/classification_report_tabnet_l3.csv")
print("✅ Saved classification report to results/classification_report_tabnet_l3.csv")

# === 混淆矩阵图 ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (TabNet L3)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/confusion_matrix_tabnet_l3.png", dpi=300)
plt.close()
print("✅ Saved confusion matrix to results/confusion_matrix_tabnet_l3.png")
