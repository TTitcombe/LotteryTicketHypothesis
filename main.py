import argparse

import torch
import torchvision
from tqdm import trange

from src import LeNet, mnist_train_transform


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_data = torchvision.datasets.MNIST(
        "data/", download=True, train=True, transform=mnist_train_transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs)

    model.train()

    for epoch in trange(args.epochs):
        correct = 0
        total = 0

        for data, targets in train_loader:
            optim.zero_grad()

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            optim.step()

            # Track performance stats
            correct += outputs.size(0)
            total += outputs.max(1)[1].eq(targets).sum().item()

        epoch_acc = 100 * total / correct

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Final accuracy: {epoch_acc:.3f}")
    print(f"Model parameters: {total_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding Lottery Tickets on an MNIST classifier")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate (default 1e-3)"
    )
    parser.add_argument("--bs", default=128, type=int, help="Batch size (default 128)")
    parser.add_argument(
        "--epochs", default=8, type=int, help="Number of epochs (default 8)"
    )

    args = parser.parse_args()

    main(args)
