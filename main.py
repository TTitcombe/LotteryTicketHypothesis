import argparse
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn.utils.prune as prune
import torchvision
from tqdm import trange

from src import LeNet, mnist_train_transform


def make_prune_permanent(model) -> torch.nn.Module:
    # TODO generalise this
    prune.remove(model.features.first_conv, "weight")
    prune.remove(model.features.second_conv, "weight")
    prune.remove(model.classifier.first_linear, "weight")
    prune.remove(model.classifier.second_linear, "weight")
    prune.remove(model.classifier.third_linear, "weight")

    return model


def prune_model(model, prune_pc) -> Tuple[torch.nn.Module, float]:
    # TODO generalise this
    parameters_to_prune = (
        (model.features.first_conv, 'weight'),
        (model.features.second_conv, 'weight'),
        (model.classifier.first_linear, 'weight'),
        (model.classifier.second_linear, 'weight'),
        (model.classifier.third_linear, 'weight'),
    )

    # Prune model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_pc,
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

    # Make pruning permanent
    #model = make_prune_permanent(model)

    return model, n_pruned_parameters

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)

    # Take initial model weights
    initial_weights = deepcopy(model.state_dict())

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_data = torchvision.datasets.MNIST(
        "data/", download=True, train=True, transform=mnist_train_transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs)

    model.train()

    # Number of parameters in total (pruned and unpruned)
    total_parameters = int(
            model.features.first_conv.weight.nelement()
            + model.features.second_conv.weight.nelement()
            + model.classifier.first_linear.weight.nelement()
            + model.classifier.second_linear.weight.nelement()
            + model.classifier.third_linear.weight.nelement()
        )

    # Percentage to *prune* per round
    prune_pc_per_round = 1 - (1 - args.prune_pc) ** (1/args.prune_rounds)

    for round in range(args.prune_rounds):
        print(f"\nPruning round {round} of {args.prune_rounds}")
        # Rest model weights

        """if round != 0:
            for parameter_name, parameter_values in model.named_parameters():
                print(parameter_name)
                non_zero_values = (parameter_values != 0)
                #initial_weights[parameter_name] *= non_zero_values"""

            #print(initial_weights)

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

        # Prune model
        model, n_pruned_parameters = prune_model(model, prune_pc_per_round)
        pruned_state_dict = model.state_dict()

        # Re-initialise lottery ticket weights
        for parameter_name, parameter_values in initial_weights.items():
            if "bias" in parameter_name:
                continue

            augmented_parameter_name = parameter_name + "_orig"
            if augmented_parameter_name in pruned_state_dict:
                pruned_state_dict[augmented_parameter_name] = parameter_values
            """weight_mask_value = pruned_state_dict[parameter_name]
            non_zero_values = (weight_mask_value != 0).float()
            print(non_zero_values.sum().item())
            pruned_weights[parameter_name] = parameter_values * non_zero_values
            non_zero_values2 = (pruned_weights[parameter_name] != 0).float()"""
        model.load_state_dict(pruned_state_dict)

        print(f"Final accuracy: {epoch_acc:.3f}")
        print(f"Model parameters: {n_pruned_parameters}/{total_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finding Lottery Tickets on an MNIST classifier")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate (default 1e-3)"
    )
    parser.add_argument("--bs", default=128, type=int, help="Batch size (default 128)")
    parser.add_argument(
        "--epochs", default=5, type=int, help="Number of epochs (default 5)"
    )
    parser.add_argument("--prune_pc", default=0.2, type=float, help="Percentage of parameters to prune over the course of the training process (default 0.2)")
    parser.add_argument("--prune_rounds", default=5, type=int, help="Number of rounds of pruning to perform (default 5)")

    args = parser.parse_args()

    main(args)
