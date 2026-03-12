import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==============================
# 自动获取项目根目录
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

y_true_path = os.path.join(DATA_DIR, "y_true_nsl_unsw.npy")
y_prob_path = os.path.join(DATA_DIR, "y_prob_nsl_unsw.npy")

# ==============================
# 加载数据
# ==============================
y_true = np.load(y_true_path)
y_prob = np.load(y_prob_path)

# ==============================
# ROC计算
# ==============================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# ==============================
# 画图
# ==============================
plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (NSL → UNSW)")
plt.legend()

plt.tight_layout()

plt.savefig("figure2_roc.png", dpi=300)

plt.show()