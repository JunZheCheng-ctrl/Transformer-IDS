# train_eval/table8_depth.py
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from models.improved_transformer import ImprovedTransformerIDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "NSL": os.path.join(BASE_DIR, "data", "processed_nsl.npz"),
    "UNSW": os.path.join(BASE_DIR, "data", "processed_unsw.npz"),
    "CIC": os.path.join(BASE_DIR, "data", "processed_cic.npz"),
}

# Table8
DEPTHS = [2, 4, 6]
SEEDS = [0, 1, 2]

MAX_TRAIN = 30000
MAX_TEST = 20000

EPOCHS = 5
BATCH_SIZE = 1024
LR = 1e-3


# ---------------------------
# Utils
# ---------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_X(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X.astype(np.float32, copy=False)


def ensure_1d_labels(y: np.ndarray, n_rows: int) -> np.ndarray:
    """
    ✅ 核心修复：把 y 强制变成 shape=[n_rows] 的 1D 标签
    绝对不允许 reshape(-1) 把 (n_rows, k) 摊平成 n_rows*k
    """
    y = np.asarray(y)

    # 先尽量把 (n,1) squeeze 成 (n,)
    if y.ndim == 2 and y.shape[1] == 1 and y.shape[0] == n_rows:
        y = y[:, 0]

    # 如果还是二维 (n,k)
    if y.ndim == 2 and y.shape[0] == n_rows and y.shape[1] > 1:
        # 如果像 one-hot（每行和≈1且全是0/1），用 argmax
        is_binary = np.all((y == 0) | (y == 1))
        row_sum_ok = np.allclose(y.sum(axis=1), 1.0)
        if is_binary and row_sum_ok:
            y = np.argmax(y, axis=1)
        else:
            # 否则取第一列作为兜底（至少保证长度正确，避免越界）
            y = y[:, 0]

    # 如果是一维但长度不对，尝试修正
    if y.ndim == 1 and y.shape[0] != n_rows:
        if y.size == n_rows:
            y = y.reshape(n_rows)
        elif y.size % n_rows == 0:
            y = y.reshape(n_rows, -1)[:, 0]
        else:
            # 最后兜底：截断或补齐
            y = y.reshape(-1)
            if y.size >= n_rows:
                y = y[:n_rows]
            else:
                pad = np.zeros((n_rows - y.size,), dtype=y.dtype)
                y = np.concatenate([y, pad], axis=0)

    # 如果还不是一维，最后强制
    if y.ndim != 1:
        y = y.reshape(n_rows)

    # 清洗 & 二值化
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    y = y.astype(np.float32, copy=False)
    y = (y > 0.5).astype(np.float32)
    return y


def load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)

    X_train = clean_X(data["X_train"])
    X_test = clean_X(data["X_test"])

    y_train = ensure_1d_labels(data["y_train"], X_train.shape[0])
    y_test = ensure_1d_labels(data["y_test"], X_test.shape[0])

    return X_train, y_train, X_test, y_test


def stratified_subsample(X, y, max_n, seed):
    """
    ✅ 采样基准必须以 X.shape[0] 为准，永远不让 idx 超界
    """
    X = np.asarray(X)
    n = X.shape[0]
    y = ensure_1d_labels(y, n)

    if (max_n is None) or (n <= max_n):
        return X, y

    uniq, cnt = np.unique(y, return_counts=True)
    if len(uniq) < 2 or np.min(cnt) < 2:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        return X[idx], y[idx]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_n, random_state=seed)
    idx, _ = next(sss.split(np.zeros((n, 1)), y))
    return X[idx], y[idx]


def align_features(X_src, X_tgt):
    d = min(X_src.shape[1], X_tgt.shape[1])
    return X_src[:, :d], X_tgt[:, :d], d


# ---------------------------
# Train / Predict (mini-batch)
# ---------------------------
def train_model(model, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    model = model.to(DEVICE)
    model.train()

    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)

        print(f"    epoch {ep}/{epochs} loss={total/n:.4f} ({time.time()-t0:.1f}s)")

    return model


@torch.no_grad()
def predict_prob(model, X, batch_size=BATCH_SIZE):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb).view(-1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        out.append(prob)
    return np.concatenate(out, axis=0)


def safe_auc(y_true, y_prob):
    y_true = ensure_1d_labels(y_true, len(y_true))
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_f1(y_true, y_prob):
    y_true = ensure_1d_labels(y_true, len(y_true))
    pred = (y_prob >= 0.5).astype(np.int32)
    return float(f1_score(y_true, pred, zero_division=0))


# ---------------------------
# Within / Transfer
# ---------------------------
def within_one_dataset(depth, seed, name, path):
    print(f"  [Within] {name} | depth={depth} seed={seed}")
    Xtr, ytr, Xte, yte = load_npz(path)

    Xtr, ytr = stratified_subsample(Xtr, ytr, MAX_TRAIN, seed)
    Xte, yte = stratified_subsample(Xte, yte, MAX_TEST, seed + 999)

    model = ImprovedTransformerIDS(d_in=Xtr.shape[1], num_layers=depth)
    model = train_model(model, Xtr, ytr)

    prob = predict_prob(model, Xte)
    auc = safe_auc(yte, prob)
    f1 = safe_f1(yte, prob)
    print(f"    -> within AUC={auc:.4f}  F1={f1:.4f}")
    return auc, f1


def transfer_one_pair(depth, seed, src_name, src_path, tgt_name, tgt_path):
    print(f"  [Transfer] {src_name}->{tgt_name} | depth={depth} seed={seed}")
    Xtr, ytr, _, _ = load_npz(src_path)
    _, _, Xte, yte = load_npz(tgt_path)

    Xtr, ytr = stratified_subsample(Xtr, ytr, MAX_TRAIN, seed)
    Xte, yte = stratified_subsample(Xte, yte, MAX_TEST, seed + 999)

    Xtr, Xte, d = align_features(Xtr, Xte)

    model = ImprovedTransformerIDS(d_in=d, num_layers=depth)
    model = train_model(model, Xtr, ytr)

    prob = predict_prob(model, Xte)
    auc = safe_auc(yte, prob)
    f1 = safe_f1(yte, prob)
    print(f"    -> transfer AUC={auc:.4f}  F1={f1:.4f}")
    return auc, f1


# ---------------------------
# Table 8
# ---------------------------
def run_table8():
    print("Device:", DEVICE)
    print("Datasets:")
    for k, v in DATASETS.items():
        print(f"  {k} -> {v}")

    pairs = [
        ("NSL", "UNSW"), ("NSL", "CIC"),
        ("UNSW", "NSL"), ("UNSW", "CIC"),
        ("CIC", "NSL"), ("CIC", "UNSW"),
    ]

    rows = []

    for depth in DEPTHS:
        print("\n" + "=" * 50)
        print(f"Running depth = {depth}")
        print("=" * 50)

        within_seed_auc, within_seed_f1 = [], []
        transfer_seed_auc, transfer_seed_f1 = [], []

        for seed in SEEDS:
            seed_all(seed)
            print(f"\nSeed = {seed}")

            # within avg over datasets
            w_auc_list, w_f1_list = [], []
            for name, path in DATASETS.items():
                a, f = within_one_dataset(depth, seed, name, path)
                w_auc_list.append(a)
                w_f1_list.append(f)

            w_auc = float(np.nanmean(w_auc_list))
            w_f1 = float(np.nanmean(w_f1_list))
            within_seed_auc.append(w_auc)
            within_seed_f1.append(w_f1)
            print(f"  => Within Avg: AUC={w_auc:.4f}  F1={w_f1:.4f}")

            # transfer avg over pairs
            t_auc_list, t_f1_list = [], []
            for s, t in pairs:
                a, f = transfer_one_pair(depth, seed, s, DATASETS[s], t, DATASETS[t])
                t_auc_list.append(a)
                t_f1_list.append(f)

            t_auc = float(np.nanmean(t_auc_list))
            t_f1 = float(np.nanmean(t_f1_list))
            transfer_seed_auc.append(t_auc)
            transfer_seed_f1.append(t_f1)
            print(f"  => Transfer Avg: AUC={t_auc:.4f}  F1={t_f1:.4f}")

        row = {
            "Depth": depth,
            "WithinAvgAUC_mean": float(np.mean(within_seed_auc)),
            "WithinAvgAUC_std": float(np.std(within_seed_auc, ddof=0)),
            "WithinAvgF1_mean": float(np.mean(within_seed_f1)),
            "WithinAvgF1_std": float(np.std(within_seed_f1, ddof=0)),
            "TransferAvgAUC_mean": float(np.mean(transfer_seed_auc)),
            "TransferAvgAUC_std": float(np.std(transfer_seed_auc, ddof=0)),
            "TransferAvgF1_mean": float(np.mean(transfer_seed_f1)),
            "TransferAvgF1_std": float(np.std(transfer_seed_f1, ddof=0)),
        }

        print("\n[Depth Summary]")
        for k, v in row.items():
            print(f"  {k}: {v:.6f}" if k != "Depth" else f"  Depth: {v}")

        rows.append(row)

    return rows


def save_csv(rows, out_path):
    cols = [
        "Depth",
        "WithinAvgAUC_mean", "WithinAvgAUC_std",
        "WithinAvgF1_mean", "WithinAvgF1_std",
        "TransferAvgAUC_mean", "TransferAvgAUC_std",
        "TransferAvgF1_mean", "TransferAvgF1_std",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join([
                str(r["Depth"]),
                f'{r["WithinAvgAUC_mean"]:.6f}', f'{r["WithinAvgAUC_std"]:.6f}',
                f'{r["WithinAvgF1_mean"]:.6f}', f'{r["WithinAvgF1_std"]:.6f}',
                f'{r["TransferAvgAUC_mean"]:.6f}', f'{r["TransferAvgAUC_std"]:.6f}',
                f'{r["TransferAvgF1_mean"]:.6f}', f'{r["TransferAvgF1_std"]:.6f}',
            ]) + "\n")
    print("\nSaved:", out_path)


if __name__ == "__main__":
    rows = run_table8()
    out_csv = os.path.join(BASE_DIR, "table8_depth.csv")
    save_csv(rows, out_csv)