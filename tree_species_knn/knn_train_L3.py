import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# === 读取数据路径 ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tree_species_transformer/data"))
X = np.load(os.path.join(base_path, "X_1440.npy"))
y = np.load(os.path.join(base_path, "y_l3_1440.npy"))
label_encoder = joblib.load(os.path.join(base_path, "label_encoder_l3_1440.pkl"))
class_names = label_encoder.classes_

# === 划分训练测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === KNN 模型训练 ===
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train, y_train)

# === 推理与评估 ===
y_pred = knn.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# === 保存结果 ===
os.makedirs("results", exist_ok=True)
joblib.dump(knn, "results/knn_l3_model.pkl")
np.save("results/y_true.npy", y_test)
np.save("results/y_pred.npy", y_pred)
np.save("results/confusion_matrix_knn_l3.npy", conf_matrix)

df_report = pd.DataFrame(report).transpose()
df_report.to_csv("results/classification_report_knn_l3.csv")
print("✅ Saved classification report to results/classification_report_knn_l3.csv")

# === 绘制混淆矩阵图 ===
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, xticklabels=class_names, yticklabels=class_names,
            annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (KNN L3)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/confusion_matrix_knn_l3.png", dpi=300)
plt.close()
print("✅ Saved confusion matrix image to results/confusion_matrix_knn_l3.png")
