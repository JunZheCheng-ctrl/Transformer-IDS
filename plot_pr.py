import sys
import os

# ===============================
# 自动找到项目根目录，并把它加入 sys.path（解决 models 导入）
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.improved_transformer import ImprovedTransformerIDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# 工具：递归找文件
# ===============================
def find_file(filename: str, root: str) -> str | None:
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def get_processed_nsl_path() -> str:
    target = "processed_nsl.npz"
    # 常见位置优先（快）
    candidates = [
        os.path.join(BASE_DIR, target),
        os.path.join(BASE_DIR, "data", target),
        os.path.join(BASE_DIR, "data", "data", target),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 全盘递归（稳）
    p = find_file(target, BASE_DIR)
    if p is not None:
        return p

    raise FileNotFoundError(
        f"找不到 {target}。\n"
        f"已检查：\n- " + "\n- ".join(candidates) +
        f"\n也递归搜索了：{BASE_DIR}\n"
        f"你可以运行：dir /s processed_nsl.npz 来确认文件实际位置。"
    )


# ===============================
# 读取数据
# ===============================
def load_dataset():
    path = get_processed_nsl_path()
    print("Loading dataset:", path)
    data = np.load(path)

    keys = set(data.files)
    if "X_train" in keys and "y_train" in keys:
        X = data["X_train"]
        y = data["y_train"]
    elif "X" in keys and "y" in keys:
        X = data["X"]
        y = data["y"]
    else:
        raise KeyError(f"{os.path.basename(path)} keys={data.files}，没有 X_train/y_train 或 X/y")

    # 统一 dtype，减少内存
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X, y


# ===============================
# 指标
# ===============================
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    return acc, prec, rec, f1, auc


# ===============================
# 输出适配：prob 或 logits
# ===============================
def to_prob(x: torch.Tensor) -> torch.Tensor:
    """
    如果输出已经在[0,1]区间，当作 prob；
    否则当作 logits 做 sigmoid。
    """
    x = x.float()
    x_min = float(x.min().detach().cpu())
    x_max = float(x.max().detach().cpu())
    if x_min < 0.0 or x_max > 1.0:
        return torch.sigmoid(x)
    return x


# ===============================
# mini-batch 训练（防 OOM）
# ===============================
def train_nn(model, X_train, y_train, X_test, epochs=3, batch_size=1024, lr=1e-3):
    model.to(device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(X_train)

    for ep in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]

            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device).view(-1)  # [B]

            optimizer.zero_grad()

            out = model(xb)
            out = out.view(-1)  # [B] 或 [B,1] -> [B]

            prob = to_prob(out)
            loss = criterion(prob, yb)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        print(f"epoch {ep+1}/{epochs} loss={total_loss:.4f}")

    # ===== eval: batch 推理防内存 =====
    model.eval()
    probs_all = []

    with torch.no_grad():
        m = len(X_test)
        for i in range(0, m, batch_size):
            xb = torch.from_numpy(X_test[i:i + batch_size]).to(device)
            out = model(xb).view(-1)
            prob = to_prob(out).detach().cpu().numpy()
            probs_all.append(prob)

    probs = np.concatenate(probs_all, axis=0).reshape(-1)
    preds = (probs > 0.5).astype(int).reshape(-1)

    return preds, probs


def main():
    # 1) load
    X, y = load_dataset()

    # 2) split（stratify 保持正负比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {}

    # =========================
    # RF（无需 mini-batch）
    # =========================
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["RF"] = compute_metrics(y_test, rf_pred, rf_prob)

    # =========================
    # CNN
    # =========================
    cnn = CNNModel(input_dim=X_train.shape[1])
    pred, prob = train_nn(cnn, X_train, y_train, X_test, epochs=3, batch_size=1024, lr=1e-3)
    results["CNN"] = compute_metrics(y_test, pred, prob)
    del cnn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================
    # LSTM（更吃内存，batch 可以再小点）
    # =========================
    lstm = LSTMModel(input_dim=X_train.shape[1])
    pred, prob = train_nn(lstm, X_train, y_train, X_test, epochs=3, batch_size=512, lr=1e-3)
    results["LSTM"] = compute_metrics(y_test, pred, prob)
    del lstm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================
    # Transformer（IFT-IDS）
    # =========================
    transformer = ImprovedTransformerIDS(d_in=X_train.shape[1])
    pred, prob = train_nn(transformer, X_train, y_train, X_test, epochs=3, batch_size=512, lr=1e-3)
    results["Transformer"] = compute_metrics(y_test, pred, prob)
    del transformer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================
    # 输出 Table 3
    # =========================
    print("\n=== Table 3 Model Comparison ===\n")
    print("Model\tAccuracy\tPrecision\tRecall\tF1\tAUC")
    for k, v in results.items():
        print(f"{k}\t{v[0]:.4f}\t{v[1]:.4f}\t{v[2]:.4f}\t{v[3]:.4f}\t{v[4]:.4f}")

    # 保存 CSV（方便直接贴进论文表）
    out_csv = os.path.join(BASE_DIR, "table3_model_comparison.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Model,Accuracy,Precision,Recall,F1,AUC\n")
        for k, v in results.items():
            f.write(f"{k},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f},{v[4]:.6f}\n")

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()