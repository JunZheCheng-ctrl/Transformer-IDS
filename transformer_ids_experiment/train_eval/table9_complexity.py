import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np


# ===============================
# 关键修复：加入项目根目录
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.improved_transformer import ImprovedTransformerIDS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 输入特征维度
INPUT_DIM = 80

# batch
BATCH_SIZE = 1024


# ===============================
# 参数量统计
# ===============================
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# 推理时间
# ===============================
def measure_inference_time(model):

    model.eval()

    x = torch.randn(BATCH_SIZE, INPUT_DIM).to(DEVICE)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)

    start = time.time()

    with torch.no_grad():
        for _ in range(100):
            model(x)

    end = time.time()

    total_time = end - start

    ms_per_sample = (total_time / 100) * 1000 / BATCH_SIZE

    return ms_per_sample


# ===============================
# 训练时间
# ===============================
def measure_train_time(model):

    model.train()

    x = torch.randn(BATCH_SIZE, INPUT_DIM).to(DEVICE)
    y = torch.randint(0, 2, (BATCH_SIZE,)).float().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = nn.BCEWithLogitsLoss()

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x).view(-1)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    start = time.time()

    for _ in range(50):

        optimizer.zero_grad()

        out = model(x).view(-1)

        loss = loss_fn(out, y)

        loss.backward()

        optimizer.step()

    end = time.time()

    sec_per_batch = (end - start) / 50

    return sec_per_batch


# ===============================
# 主函数
# ===============================
def main():

    print("\nRunning Table9 Complexity Analysis\n")

    model = ImprovedTransformerIDS(

        d_in=INPUT_DIM,
        num_layers=6

    ).to(DEVICE)

    params = count_parameters(model)

    train_time = measure_train_time(model)

    infer_time = measure_inference_time(model)

    print("\n===== Model Complexity =====")

    print("Parameters:", params)

    print("Train time per batch:", round(train_time,6),"sec")

    print("Inference time:", round(infer_time,6),"ms/sample")


if __name__ == "__main__":

    main()