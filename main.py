import argparse
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn.utils.prune as prune
import torchvision
from tqdm import trange

from src import LeNet, mnist_train_transform


def _train_model(model, optimiser, train_loader, n_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in trange(n_epochs):
        correct = 0
        total = 0

        for data, targets in train_loader:
            optimiser.zero_grad()

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            optimiser.step()

            # Track performance stats
            correct += outputs.size(0)
            total += outputs.max(1)[1].eq(targets).sum().item()

        epoch_acc = 100 * total / correct

    return model, epoch_acc


def _reinitilise_model(model, initial_weights):
    pruned_state_dict = model.state_dict()

    for parameter_name, parameter_values in initial_weights.items():
        # Pruned weights are called <parameter_name>_orig
        augmented_parameter_name = parameter_name + "_orig"

        if augmented_parameter_name in pruned_state_dict:
            pruned_state_dict[augmented_parameter_name] = parameter_values
        else:
            # Parameter name has not changed
            # e.g. bias or weights from non-pruned layer
            pruned_state_dict[parameter_name] = parameter_values

    model.load_state_dict(pruned_state_dict)

    return model


def prune_model(model, prune_pc) -> Tuple[torch.nn.Module, float]:
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
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_pc,
    )

    # TODO generalise this process
    # Number of *unpruned* parameters
    n_pruned_parameters = int(
        torch.sum(model.features.first_conv.weight != 0)
        + torch.sum(model.features.second_conv.weight != 0)
        + torch.sum(model.classifier.first_linear.weight != 0)
        + torch.sum(model.classifier.second_linear.weight != 0)
        + torch.sum(model.classifier.third_linear.weight != 0)
    )

    return model, n_pruned_parameters


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)

    # Take initial model weights
    initial_weights = deepcopy(model.state_dict())

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect training data
    train_data = torchvision.datasets.MNIST(
        "data/", download=True, train=True, transform=mnist_train_transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs)

    # Number of parameters in total (pruned and unpruned)
    total_parameters = int(
        model.features.first_conv.weight.nelement()
        + model.features.second_conv.weight.nelement()
        + model.classifier.first_linear.weight.nelement()
        + model.classifier.second_linear.weight.nelement()
        + model.classifier.third_linear.weight.nelement()
    )

    # Percentage to *prune* per round
    prune_pc_per_round = 1 - (1 - args.prune_pc) ** (1 / args.prune_rounds)

    model.train()

    for round in range(args.prune_rounds):
        print(f"\nPruning round {round} of {args.prune_rounds}")

        # Train model
        model, accuracy = _train_model(model, optim, train_loader, args.epochs, device)

        # Prune model
        model, n_pruned_parameters = prune_model(model, prune_pc_per_round)

        # Reset model
        model = _reinitilise_model(model, initial_weights)

        print(f"Model accuracy: {accuracy:.3f}%")
        print(f"New parameters: {n_pruned_parameters}/{total_parameters}")

    # Train final model
    model, accuracy = _train_model(model, optim, train_loader, args.epochs, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding Lottery Tickets on an MNIST classifier")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate (default 1e-3)"
    )
    parser.add_argument("--bs", default=128, type=int, help="Batch size (default 128)")
    parser.add_argument(
        "--epochs", default=5, type=int, help="Number of epochs (default 5)"
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
