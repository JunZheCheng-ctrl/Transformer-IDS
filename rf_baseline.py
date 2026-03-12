import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 自动获取项目根目录
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(BASE_DIR, "table_ablation.csv")
output_path = os.path.join(BASE_DIR, "figure4_ablation.png")

# ==============================
# 读取消融实验结果
# ==============================
df = pd.read_csv(csv_path)

# 如果 Variant 名字太技术化，这里顺手改成论文里更好看的版本
name_map = {
    "baseline": "Baseline",
    "auc_only": "AUC-only",
    "calibration_only": "Calibration-only",
    "full": "Full Model"
}

df["Variant"] = df["Variant"].map(name_map).fillna(df["Variant"])

# ==============================
# 画图：Mean AUC
# ==============================
plt.figure(figsize=(7, 5))

plt.bar(df["Variant"], df["MeanAUC"])

plt.ylabel("Mean AUC")
plt.title("Ablation Study Comparison")

plt.tight_layout()
plt.savefig(output_path, dpi=300)

print("Saved:", output_path)
plt.show()