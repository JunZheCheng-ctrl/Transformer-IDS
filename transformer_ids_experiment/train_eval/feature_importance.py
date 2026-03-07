import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


data = np.load("data/processed_nsl.npz")

X = data["X_train"]
y = data["y_train"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

importance = model.feature_importances_

indices = np.argsort(importance)[::-1][:20]

plt.figure(figsize=(8,6))

plt.bar(range(20), importance[indices])

plt.xlabel("Feature Index")
plt.ylabel("Importance")

plt.title("Top-20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

print("Feature importance figure saved")

plt.show()