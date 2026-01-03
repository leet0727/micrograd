import torch
import torch.nn as nn
from tqdm import trange

def main():
    torch.manual_seed(0)

    # -----------------------------
    # 1) Dataset
    # -----------------------------
    X_data = torch.tensor([
        [2.0, 4.0],
        [1.5, 3.0],
        [0.8, 2.5],
        [1.2, 1.8],
        [1.9, 3.5],
        [0.6, 1.4],
        [1.1, 2.2],
        [1.7, 1.3],
    ], dtype=torch.float32)

    Y_data = torch.tensor([
        [2.0, 4.0, 6.0, 8.0, 0.5, 16.0],
        [1.5, 3.0, 4.5, 4.5, 0.5, 3.375],
        [0.8, 2.5, 3.3, 2.0, 0.32, (0.8 ** 2.5)],
        [1.2, 1.8, 3.0, 2.16, (1.2 / 1.8), (1.2 ** 1.8)],
        [1.9, 3.5, 5.4, 6.65, (1.9 / 3.5), (1.9 ** 3.5)],
        [0.6, 1.4, 2.0, 0.84, (0.6 / 1.4), (0.6 ** 1.4)],
        [1.1, 2.2, 3.3, 2.42, 0.5, (1.1 ** 2.2)],
        [1.7, 1.3, 3.0, 2.21, (1.7 / 1.3), (1.7 ** 1.3)],
    ], dtype=torch.float32)

    n = X_data.shape[0]

    # -----------------------------
    # 2) Model
    # -----------------------------
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 6),
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # -----------------------------
    # 3) Training with progress bar
    # -----------------------------
    steps = 8000
    pbar = trange(steps, desc="Training", unit="step")

    for step in pbar:
        idx = torch.randint(0, n, (4,))
        x = X_data[idx]
        y = Y_data[idx]

        pred = net(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.6f}")

    # -----------------------------
    # 4) Test
    # -----------------------------
    with torch.no_grad():
        x_test = torch.tensor([[2.0, 4.0],
                               [1.0, 2.0],
                               [1.7, 1.3]], dtype=torch.float32)
        pred = net(x_test)

        print("\nTEST INPUTS:\n", x_test)
        print("\nPREDICTIONS:\n", pred)

        a = x_test[:, 0:1]
        b = x_test[:, 1:2]
        gt = torch.cat([a, b, a + b, a * b, a / b, a.pow(b)], dim=1)

        print("\nGROUND TRUTH:\n", gt)

if __name__ == "__main__":
    main()
