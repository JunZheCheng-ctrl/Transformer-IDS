import os
import sys
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.improved_transformer import ImprovedTransformerIDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(BASE_DIR, "data")

NSL_PATH = os.path.join(DATA_DIR, "processed_nsl.npz")


# =========================
# Load dataset
# =========================

def load_dataset():

    data = np.load(NSL_PATH)

    X = data["X_train"]
    y = data["y_train"]

    return X.astype(np.float32), y.astype(np.int32)


# =========================
# Build imbalance dataset
# =========================

def build_imbalance(X, y, attack_ratio):

    pos = X[y == 1]
    neg = X[y == 0]

    n_attack = int(len(neg) * attack_ratio / (1 - attack_ratio))

    pos = pos[:n_attack]

    X_new = np.vstack([neg, pos])
    y_new = np.array([0] * len(neg) + [1] * len(pos))

    return X_new, y_new


# =========================
# Train
# =========================

def train_model(X_train, y_train, X_val, y_val):

    model = ImprovedTransformerIDS(
        d_in=X_train.shape[1],
        use_calibration=True
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.BCEWithLogitsLoss()

    X_train = torch.tensor(X_train).to(DEVICE)
    y_train = torch.tensor(y_train).float().to(DEVICE)

    X_val = torch.tensor(X_val).to(DEVICE)

    batch = 512

    for epoch in range(2):

        perm = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch):

            idx = perm[i:i+batch]

            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            logits = model(xb)

            loss = loss_fn(logits, yb)

            loss.backward()

            optimizer.step()

    model.eval()

    with torch.no_grad():

        prob = torch.sigmoid(model(X_val)).cpu().numpy()

    auc = roc_auc_score(y_val, prob)

    pred = (prob > 0.5).astype(int)

    f1 = f1_score(y_val, pred)
    rec = recall_score(y_val, pred)

    return auc, f1, rec


# =========================
# Main experiment
# =========================

def main():

    ratios = [0.5, 0.2, 0.1, 0.05]

    X, y = load_dataset()

    results = []

    for r in ratios:

        print("\nAttack ratio:", r)

        X_new, y_new = build_imbalance(X, y, r)

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

        auc, f1, rec = train_model(X_train, y_train, X_test, y_test)

        print("AUC:", auc)
        print("F1 :", f1)
        print("Recall:", rec)

        results.append((r, auc, f1, rec))

    out = os.path.join(BASE_DIR, "table_imbalance.csv")

    with open(out, "w") as f:

        f.write("AttackRatio,AUC,F1,Recall\n")

        for r in results:

            f.write(",".join(map(str, r)) + "\n")

    print("\nSaved:", out)


if __name__ == "__main__":
    main()