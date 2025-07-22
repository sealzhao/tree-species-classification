import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# === Load data ===
# ==== 加载标签 ====
y_true = np.load("y_l3_true.npy")
y_pred = np.load("y_l3_preds.npy")

# ==== 类别名 ====
label_encoder = joblib.load("data/label_encoder_l3_1440.pkl")
class_names = label_encoder.classes_

# ==== 混淆矩阵 ====
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

# 避免除以 0
cm_sum = cm.sum(axis=1, keepdims=True)
cm_sum[cm_sum == 0] = 1
cm_norm = cm.astype('float') / cm_sum

# 分类报告，避免缺失键
report = classification_report(y_true, y_pred, labels=range(len(class_names)), output_dict=True, zero_division=0)
f1_scores = [report.get(str(i), {}).get('f1-score', 0.0) for i in range(len(class_names))]

# ==== F1 柱状图 ====
plt.figure(figsize=(14, 6))
plt.bar(class_names, f1_scores, color=plt.cm.viridis(np.linspace(0, 1, len(class_names))))
plt.xticks(rotation=90)
plt.title("F1 Score per Class (L3)")
plt.tight_layout()
plt.savefig("plots/f1_score_bar_chart_l3.png", dpi=300)
plt.close()

# ==== 归一化混淆矩阵图 ====
plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, annot=False, cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix (L3)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/confusion_matrix_normalized_l3.png", dpi=300)
plt.close()
