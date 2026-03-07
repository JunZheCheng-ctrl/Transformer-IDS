import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

from data.preprocess_nslkdd import X_train, X_test, y_train, y_test
from models.improved_transformer import ImprovedTransformerIDS
from train_eval.auc_surrogate import PairwiseAUCLoss


# ==============================
# 固定随机种子
# ==============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================
# 评估函数
# ==============================
def evaluate(model, loader, device="cpu"):

    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())

    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy()

    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    pre = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    mcc = matthews_corrcoef(y, pred)

    return acc, pre, rec, f1, auc, ap, mcc


# ==============================
# 单次训练
# ==============================
def main(seed=0, use_auc_loss=True):

    set_seed(seed)

    device = "cpu"

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    ytr = torch.tensor(y_train.values, dtype=torch.float32)
    yte = torch.tensor(y_test.values, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    d_in = Xtr.shape[1]

    model = ImprovedTransformerIDS(d_in=d_in).to(device)

    bce = nn.BCEWithLogitsLoss()

    auc_loss = PairwiseAUCLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    epochs = 10

    lam_auc = 0.2

    # ==========================
    # 训练
    # ==========================
    for ep in range(epochs):

        model.train()

        total = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = bce(logits, yb)

            if use_auc_loss:
                loss = loss + lam_auc * auc_loss(logits, yb)

            loss.backward()

            optimizer.step()

            total += loss.item()

        print(f"epoch {ep} loss {total/len(train_loader):.6f}")

    # ==========================
    # 测试
    # ==========================
    acc, pre, rec, f1, auc, ap, mcc = evaluate(model, test_loader, device)

    print("\n===== IFT Evaluation Results =====")

    print("Accuracy:", acc)
    print("Precision:", pre)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)
    print("AP:", ap)
    print("MCC:", mcc)

    return [acc, pre, rec, f1, auc, ap, mcc]


# ==============================
# 多seed实验
# ==============================
if __name__ == "__main__":

    seeds = [0,1,2,3,4]

    results = []

    for s in seeds:

        print("\n======================")
        print("Running seed:", s)
        print("======================")

        r = main(seed=s, use_auc_loss=True)

        results.append(r)

    results = np.array(results)

    mean = results.mean(axis=0)
    std = results.std(axis=0)

    print("\n===== FINAL RESULTS (mean ± std) =====")

    names = ["Acc","Pre","Rec","F1","AUC","AP","MCC"]

    for i,n in enumerate(names):

        print(f"{n}: {mean[i]:.4f} ± {std[i]:.4f}")