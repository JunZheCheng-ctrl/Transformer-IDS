import numpy as np
from sklearn.metrics import f1_score

def search_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1