import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ================================
# 加载 npz
# ================================
def load_dataset(path):

    data = np.load(path)

    print("Dataset keys:", data.files)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_test, y_test


# ================================
# 特征对齐
# ================================
def align_features(X1, X2):

    min_features = min(X1.shape[1], X2.shape[1])

    X1 = X1[:, :min_features]
    X2 = X2[:, :min_features]

    return X1, X2


# ================================
# 样本数量对齐
# ================================
def align_samples(X, y):

    min_len = min(len(X), len(y))

    X = X[:min_len]
    y = y[:min_len]

    return X, y


# ================================
# 单个实验
# ================================
def run_experiment(train_path, test_path, name):

    print("\n==============================")
    print(name)
    print("==============================")

    X_train, y_train, _, _ = load_dataset(train_path)
    _, _, X_test, y_test = load_dataset(test_path)

    print("Before align:", X_train.shape, X_test.shape)

    X_train, X_test = align_features(X_train, X_test)

    print("After align :", X_train.shape, X_test.shape)

    X_test, y_test = align_samples(X_test, y_test)

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nResults")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("AUC      :", round(auc, 4))


# ================================
# 主程序
# ================================
def main():

    nsl = "data/processed_nsl.npz"
    unsw = "data/processed_unsw.npz"
    cic = "data/processed_cic.npz"

    run_experiment(nsl, unsw, "Train NSL → Test UNSW")
    run_experiment(nsl, cic, "Train NSL → Test CIC")

    run_experiment(unsw, nsl, "Train UNSW → Test NSL")
    run_experiment(unsw, cic, "Train UNSW → Test CIC")

    run_experiment(cic, nsl, "Train CIC → Test NSL")
    run_experiment(cic, unsw, "Train CIC → Test UNSW")

    print("\nAll experiments finished")


if __name__ == "__main__":
    main()