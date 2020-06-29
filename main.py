import argparse

import torch
import torch.nn.utils.prune as prune
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

    # Percentage to *prune* per round
    prune_pc_per_round = 1 - (1 - args.prune_pc) ** (1 / args.prune_rounds)

    for round in range(args.prune_rounds):
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

        # TODO generalise this
        parameters_to_prune = (
            (model.features.first_conv, "weight"),
            (model.features.second_conv, "weight"),
            (model.classifier.first_linear, "weight"),
            (model.classifier.second_linear, "weight"),
            (model.classifier.third_linear, "weight"),
        )

        # Prune model
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_pc_per_round,
        )

        # TODO re-initialise lottery ticket weights

        # TODO generalise this process
        # Number of *unpruned* parameters
        pruned_parameters = float(
            torch.sum(model.features.first_conv.weight != 0)
            + torch.sum(model.features.second_conv.weight != 0)
            + torch.sum(model.classifier.first_linear.weight != 0)
            + torch.sum(model.classifier.second_linear.weight != 0)
            + torch.sum(model.classifier.third_linear.weight != 0)
        )

        # Number of parameters in total (pruned and unpruned)
        total_parameters = float(
            model.features.first_conv.weight.nelement()
            + model.features.second_conv.weight.nelement()
            + model.classifier.first_linear.weight.nelement()
            + model.classifier.second_linear.weight.nelement()
            + model.classifier.third_linear.weight.nelement()
        )

        print(f"Final accuracy: {epoch_acc:.3f}")
        print(f"Model parameters: {pruned_parameters}")
        print(f"Total parameters: {total_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding Lottery Tickets on an MNIST classifier")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate (default 1e-3)"
    )
    parser.add_argument("--bs", default=128, type=int, help="Batch size (default 128)")
    parser.add_argument(
        "--epochs", default=8, type=int, help="Number of epochs (default 8)"
    )
    parser.add_argument(
        "--prune_pc",
        default=0.2,
        type=float,
        help="Percentage of parameters to prune over the course of the training process (default 0.2)",
    )
    parser.add_argument(
        "--prune_rounds",
        default=5,
        type=int,
        help="Number of rounds of pruning to perform (default 5)",
    )

    args = parser.parse_args()

    main(args)
