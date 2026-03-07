import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc


def load_dataset(path):

    data = np.load(path)

    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def align_features(X1, X2):

    min_features = min(X1.shape[1], X2.shape[1])

    return X1[:, :min_features], X2[:, :min_features]


def run_pr(train_path, test_path, label):

    X_train, y_train, _, _ = load_dataset(train_path)
    _, _, X_test, y_test = load_dataset(test_path)

    X_train, X_test = align_features(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]

    min_len = min(len(y_test), len(y_prob))

    y_test = y_test[:min_len]
    y_prob = y_prob[:min_len]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall, precision)

    return recall, precision, pr_auc, label


def main():

    datasets = {

        "NSL": "data/processed_nsl.npz",
        "UNSW": "data/processed_unsw.npz",
        "CIC": "data/processed_cic.npz"
    }

    experiments = [

        ("NSL","UNSW"),
        ("NSL","CIC"),
        ("UNSW","NSL"),
        ("UNSW","CIC"),
        ("CIC","NSL"),
        ("CIC","UNSW")
    ]

    plt.figure(figsize=(7,6))

    for train,test in experiments:

        recall,precision,pr_auc,label = run_pr(
            datasets[train],
            datasets[test],
            f"{train}→{test}"
        )

        plt.plot(
            recall,
            precision,
            label=f"{label} (AUC={pr_auc:.2f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-Dataset Precision-Recall Curve")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "pr_cross_dataset.png",
        dpi=300
    )

    print("PR figure saved as pr_cross_dataset.png")

    plt.show()


if __name__ == "__main__":
    main()