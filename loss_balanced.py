import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# =================================
# 项目路径
# =================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

NSL_PATH = os.path.join(DATA_DIR, "processed_nsl.npz")
UNSW_PATH = os.path.join(DATA_DIR, "processed_unsw.npz")
CIC_PATH = os.path.join(DATA_DIR, "processed_cic.npz")

OUTPUT_PATH = os.path.join(BASE_DIR, "figure5_feature_importance.png")


# =================================
# 网络流特征名称（41维）
# =================================
FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


# =================================
# 加载数据
# =================================
def load_dataset(path):

    print("Loading:", path)

    data = np.load(path)

    X = np.concatenate([data["X_train"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])

    print("Samples:", len(X), "Feature dim:", X.shape[1])

    return X, y


# =================================
# 特征对齐
# =================================
def align_features(X1, X2, X3):

    min_dim = min(X1.shape[1], X2.shape[1], X3.shape[1])

    X1 = X1[:, :min_dim]
    X2 = X2[:, :min_dim]
    X3 = X3[:, :min_dim]

    print("Aligned feature dim:", min_dim)

    return X1, X2, X3


# =================================
# 样本对齐
# =================================
def align_samples(X, y):

    min_len = min(len(X), len(y))

    X = X[:min_len]
    y = y[:min_len]

    return X, y


# =================================
# 主函数
# =================================
def main():

    print("\nLoading datasets...\n")

    X1, y1 = load_dataset(NSL_PATH)
    X2, y2 = load_dataset(UNSW_PATH)
    X3, y3 = load_dataset(CIC_PATH)

    # 特征对齐
    X1, X2, X3 = align_features(X1, X2, X3)

    # 样本对齐
    X1, y1 = align_samples(X1, y1)
    X2, y2 = align_samples(X2, y2)
    X3, y3 = align_samples(X3, y3)

    # 合并数据
    X = np.concatenate([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    X, y = align_samples(X, y)

    print("\nMerged dataset")
    print("Total samples:", len(X))
    print("Feature dim:", X.shape[1])

    # =================================
    # Random Forest
    # =================================
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X, y)

    importances = model.feature_importances_

    # =================================
    # Top 10 特征
    # =================================
    indices = np.argsort(importances)[::-1][:10]

    values = importances[indices]

    labels = [FEATURE_NAMES[i] for i in indices]

    print("\nTop 10 Important Features:")

    for i in range(len(labels)):

        print(labels[i], ":", round(values[i], 4))

    # =================================
    # 绘图
    # =================================
    plt.figure(figsize=(7,5))

    plt.barh(labels[::-1], values[::-1], edgecolor="black")

    plt.xlabel("Feature Importance")

    plt.title("Top 10 Important Network Features")

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300)

    print("\nSaved:", OUTPUT_PATH)

    plt.show()


# =================================
# 程序入口
# =================================
if __name__ == "__main__":
    main()