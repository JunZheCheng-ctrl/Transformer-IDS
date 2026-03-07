import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from models.cnn_model import CNNModel
from models.cnn_lstm_model import CNNLSTMModel
from models.improved_transformer import ImprovedTransformerIDS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "N": "data/processed_nsl.npz",
    "U": "data/processed_unsw.npz",
    "C": "data/processed_cic.npz"
}


# ===============================
# Load dataset
# ===============================
def load_dataset(path):

    data = np.load(path)

    if "X_train" in data:
        X = data["X_train"]
        y = data["y_train"]
    else:
        X = data["X"]
        y = data["y"]

    return X.astype(np.float32), y.astype(np.float32)


# ===============================
# Metrics
# ===============================
def compute_metrics(y_true, prob):

    pred = (prob > 0.5).astype(int)

    auc = roc_auc_score(y_true, prob)
    f1 = f1_score(y_true, pred)

    return auc, f1


# ===============================
# RF
# ===============================
def train_rf(X, y):

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


# ===============================
# Train NN
# ===============================
def train_nn(model, X, y, epochs=3, batch=256):

    model.to(DEVICE)

    X = torch.tensor(X).to(DEVICE)
    y = torch.tensor(y).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for e in range(epochs):

        perm = torch.randperm(len(X))

        for i in range(0, len(X), batch):

            idx = perm[i:i+batch]

            xb = X[idx]
            yb = y[idx]

            optimizer.zero_grad()

            out = model(xb).view(-1)
            prob = torch.sigmoid(out)

            loss = loss_fn(prob, yb)

            loss.backward()
            optimizer.step()

        print("epoch", e+1, "done")

    return model


# ===============================
# Batch inference (解决OOM)
# ===============================
def predict_nn(model, X, batch=512):

    model.eval()

    probs = []

    with torch.no_grad():

        for i in range(0, len(X), batch):

            xb = torch.tensor(X[i:i+batch]).to(DEVICE)

            prob = torch.sigmoid(model(xb).view(-1))

            probs.append(prob.cpu().numpy())

    return np.concatenate(probs)


# ===============================
# Experiment
# ===============================
def run_experiment():

    auc_table = {}
    f1_table = {}

    for train_key, train_path in DATASETS.items():

        X_train, y_train = load_dataset(train_path)

        print("\nTraining on", train_key)

        for test_key, test_path in DATASETS.items():

            if train_key == test_key:
                continue

            X_test, y_test = load_dataset(test_path)

            name = f"{train_key}->{test_key}"

            print("Testing", name)

            # feature alignment
            min_dim = min(X_train.shape[1], X_test.shape[1])

            X_train_use = X_train[:, :min_dim]
            X_test_use = X_test[:, :min_dim]

            # ================= RF =================

            rf = train_rf(X_train_use, y_train)

            prob = rf.predict_proba(X_test_use)[:,1]

            auc, f1 = compute_metrics(y_test, prob)

            auc_table.setdefault("RF", {})[name] = auc
            f1_table.setdefault("RF", {})[name] = f1

            # ================= CNN =================

            cnn = train_nn(
                CNNModel(min_dim),
                X_train_use,
                y_train
            )

            prob = predict_nn(cnn, X_test_use)

            auc, f1 = compute_metrics(y_test, prob)

            auc_table.setdefault("CNN", {})[name] = auc
            f1_table.setdefault("CNN", {})[name] = f1

            # ================= CNN-LSTM =================

            cnn_lstm = train_nn(
                CNNLSTMModel(min_dim),
                X_train_use,
                y_train
            )

            prob = predict_nn(cnn_lstm, X_test_use)

            auc, f1 = compute_metrics(y_test, prob)

            auc_table.setdefault("CNN-LSTM", {})[name] = auc
            f1_table.setdefault("CNN-LSTM", {})[name] = f1

            # ================= Transformer =================

            trans = train_nn(
                ImprovedTransformerIDS(min_dim),
                X_train_use,
                y_train
            )

            prob = predict_nn(trans, X_test_use)

            auc, f1 = compute_metrics(y_test, prob)

            auc_table.setdefault("Transformer", {})[name] = auc
            f1_table.setdefault("Transformer", {})[name] = f1

    return auc_table, f1_table


# ===============================
# Save CSV
# ===============================
def save_table(table, filename):

    keys = list(next(iter(table.values())).keys())

    with open(filename, "w") as f:

        f.write("Model," + ",".join(keys) + "\n")

        for m in table:

            row = [str(table[m][k]) for k in keys]

            f.write(m + "," + ",".join(row) + "\n")


# ===============================
# main
# ===============================
if __name__ == "__main__":

    auc, f1 = run_experiment()

    save_table(auc, "table4_auc.csv")
    save_table(f1, "table5_f1.csv")

    print("\nSaved table4_auc.csv and table5_f1.csv")