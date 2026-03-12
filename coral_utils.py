import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.improved_transformer import ImprovedTransformerIDS
from train_eval.auc_surrogate import PairwiseAUCLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(BASE_DIR, "data")

NSL_PATH = os.path.join(DATA_DIR, "processed_nsl.npz")
UNSW_PATH = os.path.join(DATA_DIR, "processed_unsw.npz")
CIC_PATH = os.path.join(DATA_DIR, "processed_cic.npz")


# =========================
# Load dataset
# =========================
def load_dataset(path):

    print("Loading:", path)

    data = np.load(path)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    return X_train, y_train, X_test, y_test


# =========================
# Feature alignment
# =========================
def align_features(X1, X2):

    min_dim = min(X1.shape[1], X2.shape[1])

    return X1[:, :min_dim], X2[:, :min_dim]


# =========================
# Threshold search
# =========================
def search_best_threshold(y_true, prob):

    best_t = 0.5
    best_f1 = -1

    for t in np.arange(0.05, 0.96, 0.05):

        pred = (prob >= t).astype(int)

        f1 = f1_score(y_true, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


# =========================
# Build model
# =========================
def build_model(variant, d_in):

    if variant == "baseline":

        return ImprovedTransformerIDS(
            d_in=d_in,
            use_calibration=False
        )

    if variant == "auc_only":

        return ImprovedTransformerIDS(
            d_in=d_in,
            use_calibration=False
        )

    if variant == "calibration_only":

        return ImprovedTransformerIDS(
            d_in=d_in,
            use_calibration=True
        )

    if variant == "full":

        return ImprovedTransformerIDS(
            d_in=d_in,
            use_calibration=True
        )


# =========================
# Train model
# =========================
def train_model(model, X_train, y_train, X_val, y_val, variant):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCEWithLogitsLoss()

    auc_loss = PairwiseAUCLoss()

    X_train = torch.tensor(X_train).to(DEVICE)
    y_train = torch.tensor(y_train).to(DEVICE)

    X_val = torch.tensor(X_val).to(DEVICE)

    best_auc = -1
    best_state = None

    batch = 512

    for epoch in range(3):

        model.train()

        perm = torch.randperm(len(X_train))

        total_loss = 0

        for i in range(0, len(X_train), batch):

            idx = perm[i:i + batch]

            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            logits = model(xb)

            loss = bce(logits, yb)

            if variant in ["auc_only", "full"]:
                loss += 0.2 * auc_loss(logits, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        with torch.no_grad():

            val_prob = torch.sigmoid(model(X_val)).cpu().numpy()

        val_auc = roc_auc_score(y_val, val_prob)

        print(f"[{variant}] epoch {epoch+1}/3 loss={total_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:

            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return model


# =========================
# Predict (batch safe)
# =========================
def predict(model, X):

    model.eval()

    batch = 4096

    probs = []

    with torch.no_grad():

        for i in range(0, len(X), batch):

            xb = torch.tensor(X[i:i + batch]).to(DEVICE)

            logits = model(xb)

            prob = torch.sigmoid(logits).cpu().numpy()

            probs.append(prob)

    prob = np.concatenate(probs)

    return prob


# =========================
# Evaluate
# =========================
def evaluate_model(model, X_test, y_test, threshold):

    prob = predict(model, X_test)

    # 关键修复
    min_len = min(len(prob), len(y_test))

    prob = prob[:min_len]
    y_test = y_test[:min_len]

    pred = (prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, prob)

    f1 = f1_score(y_test, pred, zero_division=0)

    return auc, f1


# =========================
# Run transfer
# =========================
def run_transfer(train_path, test_path, name, variant):

    print("\n==============================")
    print(name, "| variant =", variant)
    print("==============================")

    X_train, y_train, _, _ = load_dataset(train_path)

    _, _, X_test, y_test = load_dataset(test_path)

    X_train, X_test = align_features(X_train, X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42
    )

    model = build_model(variant, X_train.shape[1])

    model = train_model(model, X_train, y_train, X_val, y_val, variant)

    val_prob = predict(model, X_val)

    threshold = search_best_threshold(y_val, val_prob)

    auc, f1 = evaluate_model(model, X_test, y_test, threshold)

    print("test_auc:", auc)
    print("test_f1 :", f1)

    return auc, f1


# =========================
# Main
# =========================
def main():

    transfers = [

        ("N→U", NSL_PATH, UNSW_PATH),
        ("U→N", UNSW_PATH, NSL_PATH),
        ("N→C", NSL_PATH, CIC_PATH),

    ]

    variants = [

        "baseline",
        "auc_only",
        "calibration_only",
        "full"

    ]

    results = []

    for variant in variants:

        aucs = []
        f1s = []

        for name, train_path, test_path in transfers:

            auc, f1 = run_transfer(train_path, test_path, name, variant)

            aucs.append(auc)
            f1s.append(f1)

        results.append((
            variant,
            aucs[0], aucs[1], aucs[2], np.mean(aucs),
            f1s[0], f1s[1], f1s[2], np.mean(f1s)
        ))

    output = os.path.join(BASE_DIR, "table_ablation.csv")

    with open(output, "w") as f:

        f.write("Variant,AUC_NU,AUC_UN,AUC_NC,MeanAUC,F1_NU,F1_UN,F1_NC,MeanF1\n")

        for r in results:

            f.write(",".join(map(str, r)) + "\n")

    print("\nSaved:", output)


if __name__ == "__main__":
    main()