import sys
import os

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


# ==============================
# Load dataset
# ==============================
def load_dataset(path):

    data = np.load(path)

    if "X_train" in data:
        X = data["X_train"]
        y = data["y_train"]
    else:
        X = data["X"]
        y = data["y"]

    return X.astype(np.float32), y.astype(np.float32)


# ==============================
# Metrics
# ==============================
def compute_metrics(y_true, prob):

    pred = (prob > 0.5).astype(int)

    auc = roc_auc_score(y_true, prob)
    f1 = f1_score(y_true, pred)

    return auc, f1


# ==============================
# Train model
# ==============================
def train(model, X, y, epochs=3, batch=256):

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

        print("epoch", e+1)

    return model


# ==============================
# Predict (batch)
# ==============================
def predict(model, X, batch=512):

    model.eval()

    probs = []

    with torch.no_grad():

        for i in range(0, len(X), batch):

            xb = torch.tensor(X[i:i+batch]).to(DEVICE)

            prob = torch.sigmoid(model(xb).view(-1))

            probs.append(prob.cpu().numpy())

    return np.concatenate(probs)


# ==============================
# LODO experiment
# ==============================
def run():

    table = {}

    for test_key in DATASETS:

        print("\nLODO Test:", test_key)

        X_train_list = []
        y_train_list = []

        # collect training datasets
        for k in DATASETS:

            if k == test_key:
                continue

            X, y = load_dataset(DATASETS[k])

            X_train_list.append(X)
            y_train_list.append(y)

        X_test, y_test = load_dataset(DATASETS[test_key])

        # ==========================
        # feature alignment
        # ==========================

        feature_dims = [x.shape[1] for x in X_train_list]
        feature_dims.append(X_test.shape[1])

        min_dim = min(feature_dims)

        X_train_list = [x[:, :min_dim] for x in X_train_list]

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        X_test = X_test[:, :min_dim]

        print("Feature dim:", min_dim)

        # ==========================
        # train model
        # ==========================

        model = ImprovedTransformerIDS(min_dim)

        model = train(model, X_train, y_train)

        prob = predict(model, X_test)

        auc, f1 = compute_metrics(y_test, prob)

        table[test_key] = [auc, f1]

        print("AUC:", auc, "F1:", f1)

    return table


# ==============================
# Save CSV
# ==============================
def save(table):

    with open("table6_lodo.csv", "w") as f:

        f.write("TestDataset,AUC,F1\n")

        for k in table:

            auc, f1 = table[k]

            f.write(f"{k},{auc},{f1}\n")


# ==============================
# main
# ==============================
if __name__ == "__main__":

    table = run()

    save(table)

    print("\nSaved table6_lodo.csv")