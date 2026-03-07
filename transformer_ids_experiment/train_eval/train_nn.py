import torch
import numpy as np
from models.transformer_model import TransformerIDS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.cnn_lstm_model import CNNLSTMModel

from data.preprocess_nslkdd import X_train, X_test, y_train, y_test


# =========================
# 设备
# =========================

device = "cpu"


# =========================
# 数据转换
# =========================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)


# =========================
# 选择模型
# =========================

# Transformer
model = TransformerIDS(X_train.shape[1]).to(device)



# =========================
# 训练配置
# =========================

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10
batch_size = 256


# =========================
# 训练
# =========================

for epoch in range(epochs):

    model.train()

    perm = torch.randperm(X_train.size(0))

    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):

        idx = perm[i:i + batch_size]

        batch_x = X_train[idx]
        batch_y = y_train[idx]

        pred = model(batch_x).squeeze()

        loss = criterion(pred, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("epoch", epoch, "loss", total_loss)


# =========================
# 测试
# =========================

model.eval()

with torch.no_grad():

    prob = model(X_test).squeeze().numpy()

pred = (prob > 0.5).astype(int)

y_true = y_test.numpy()


# =========================
# 评价指标
# =========================

acc = accuracy_score(y_true, pred)

precision = precision_score(y_true, pred)

recall = recall_score(y_true, pred)

f1 = f1_score(y_true, pred)

auc = roc_auc_score(y_true, prob)

ap = average_precision_score(y_true, prob)

mcc = matthews_corrcoef(y_true, pred)


# =========================
# 输出
# =========================

print("\n===== Evaluation Results =====")

print("Accuracy:", acc)

print("Precision:", precision)

print("Recall:", recall)

print("F1:", f1)

print("AUC:", auc)

print("AP:", ap)

print("MCC:", mcc)