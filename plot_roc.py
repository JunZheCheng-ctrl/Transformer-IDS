import torch
import time
import numpy as np

from models.improved_transformer import ImprovedTransformerIDS


# =================================
# 参数统计
# =================================
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =================================
# 主程序
# =================================
def main():

    feature_dim = 41

    model = ImprovedTransformerIDS(feature_dim)

    params = count_parameters(model)

    print("Trainable parameters:", params)


    # =================================
    # 训练时间测试
    # =================================

    batch_size = 256

    x = torch.randn(batch_size, feature_dim)

    y = torch.randint(0,2,(batch_size,)).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    start = time.time()

    for _ in range(50):

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_fn(logits, y)

        loss.backward()

        optimizer.step()

    end = time.time()

    train_time = (end - start) / 50

    print("Train time per batch:", train_time)


    # =================================
    # 推理时间
    # =================================

    start = time.time()

    for _ in range(1000):

        with torch.no_grad():

            model(x)

    end = time.time()

    infer_time = (end - start) / (1000 * batch_size)

    infer_time_ms = infer_time * 1000

    print("Inference time per sample (ms):", infer_time_ms)


if __name__ == "__main__":

    main()