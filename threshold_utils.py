import sys
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, f1_score

from models.improved_transformer import ImprovedTransformerIDS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "N": "data/processed_nsl.npz",
    "U": "data/processed_unsw.npz",
    "C": "data/processed_cic.npz"
}


# ======================
# Load dataset
# ======================

def load_dataset(path):

    data = np.load(path)

    X = data["X_train"].astype(np.float32)
    y = data["y_train"].astype(np.float32)

    # 为了加速，只取一部分
    X = X[:20000]
    y = y[:20000]

    return X, y


# ======================
# Train model
# ======================

def train(model, X, y):

    model.to(DEVICE)

    X = torch.tensor(X).to(DEVICE)
    y = torch.tensor(y).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(2):

        optimizer.zero_grad()

        out = model(X).view(-1)

        prob = torch.sigmoid(out)

        loss = torch.nn.functional.binary_cross_entropy(prob, y)

        loss.backward()

        optimizer.step()

    return model


# ======================
# Run experiment
# ======================

def run():

    variants = {

        "Full": lambda d: ImprovedTransformerIDS(d),

        "NoCalibration": lambda d: ImprovedTransformerIDS(d, use_calibration=False),

        "NoGate": lambda d: ImprovedTransformerIDS(d),

    }

    table = {}

    for name in variants:

        print("\nRunning:", name)

        auc_list = []
        f1_list = []

        for k in DATASETS:

            X, y = load_dataset(DATASETS[k])

            model = variants[name](X.shape[1])

            model = train(model, X, y)

            with torch.no_grad():

                prob = torch.sigmoid(
                    model(torch.tensor(X).to(DEVICE)).view(-1)
                ).cpu().numpy()

            pred = (prob > 0.5)

            auc = roc_auc_score(y, prob)
            f1 = f1_score(y, pred)

            auc_list.append(auc)
            f1_list.append(f1)

        table[name] = [np.mean(auc_list), np.mean(f1_list)]

    return table


# ======================
# Save table
# ======================

def save(table):

    with open("table7_ablation.csv", "w") as f:

        f.write("Variant,AUC,F1\n")

        for k in table:

            auc, f1 = table[k]

            f.write(f"{k},{auc},{f1}\n")


# ======================
# main
# ======================

if __name__ == "__main__":

    table = run()

    save(table)

    print("\nSaved table7_ablation.csv")