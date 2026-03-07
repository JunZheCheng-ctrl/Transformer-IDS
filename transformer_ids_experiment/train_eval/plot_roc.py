import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


def load_dataset(path):

    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_test, y_test


def align_features(X1, X2):

    min_features = min(X1.shape[1], X2.shape[1])

    return X1[:, :min_features], X2[:, :min_features]


def run_roc(train_path, test_path, label):

    print(f"\nRunning ROC experiment: {label}")

    X_train, y_train, _, _ = load_dataset(train_path)
    _, _, X_test, y_test = load_dataset(test_path)

    print("Before align:", X_train.shape, X_test.shape)

    X_train, X_test = align_features(X_train, X_test)

    print("After align :", X_train.shape, X_test.shape)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 防止标签长度不一致
    min_len = min(len(y_test), len(y_prob))
    y_test = y_test[:min_len]
    y_prob = y_prob[:min_len]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    print("AUC:", roc_auc)

    return fpr, tpr, roc_auc, label


def main():

    datasets = {
        "NSL": "data/processed_nsl.npz",
        "UNSW": "data/processed_unsw.npz",
        "CIC": "data/processed_cic.npz"
    }

    experiments = [

        ("NSL", "UNSW"),
        ("NSL", "CIC"),

        ("UNSW", "NSL"),
        ("UNSW", "CIC"),

        ("CIC", "NSL"),
        ("CIC", "UNSW")
    ]

    plt.figure(figsize=(7, 6))

    for train, test in experiments:

        fpr, tpr, roc_auc, label = run_roc(
            datasets[train],
            datasets[test],
            f"{train}→{test}"
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{label} (AUC={roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Dataset ROC Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "roc_cross_dataset.png",
        dpi=300
    )

    print("\nROC figure saved as roc_cross_dataset.png")

    plt.show()


if __name__ == "__main__":
    main()