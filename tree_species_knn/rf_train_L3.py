import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === 设置 transformer 项目的数据路径 ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tree_species_transformer/data"))

# === 加载数据 ===
X = np.load(os.path.join(base_path, "X_1440.npy"))
y = np.load(os.path.join(base_path, "y_l3_1440.npy"))
label_encoder = joblib.load(os.path.join(base_path, "label_encoder_l3_1440.pkl"))
class_names = label_encoder.classes_

# === 分割训练集和验证集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 训练 Random Forest 模型 ===
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# === 推理与评估 ===
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# === 保存模型和评估数据 ===
os.makedirs("results_rf", exist_ok=True)
joblib.dump(rf, "results_rf/rf_l3_model.pkl")
np.save("results_rf/y_true_rf.npy", y_test)
np.save("results_rf/y_pred_rf.npy", y_pred)
pd.DataFrame(report).transpose().to_csv("results_rf/classification_report_rf_l3.csv")
print("✅ Saved classification report to results_rf/classification_report_rf_l3.csv")

# === 绘制并保存混淆矩阵图 ===
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, xticklabels=class_names, yticklabels=class_names,
            annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Random Forest L3)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_rf/confusion_matrix_rf_l3.png", dpi=300)
plt.close()
print("✅ Saved confusion matrix to results_rf/confusion_matrix_rf_l3.png")
