import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc


# =================================
# 自动获取项目根目录
# =================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

y_true_path = os.path.join(DATA_DIR, "y_true_nsl_unsw.npy")
y_prob_path = os.path.join(DATA_DIR, "y_prob_nsl_unsw.npy")


# =================================
# 加载数据
# =================================
print("Loading data...")

y_true = np.load(y_true_path)
y_prob = np.load(y_prob_path)

print("Samples:", len(y_true))


# =================================
# 计算 PR 曲线
# =================================
precision, recall, _ = precision_recall_curve(y_true, y_prob)

pr_auc = auc(recall, precision)

print("PR-AUC:", round(pr_auc, 4))


# =================================
# 绘图
# =================================
plt.figure(figsize=(6,5))

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AUC={pr_auc:.3f})",
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision–Recall Curve (NSL → UNSW)")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


# =================================
# 保存图片
# =================================
output_path = os.path.join(BASE_DIR, "figure3_pr.png")

plt.savefig(output_path, dpi=300)

print("Saved:", output_path)

plt.show()