from collections import OrderedDict

from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("first_conv", nn.Conv2d(1, 6, 3)),
                    ("first_relu", nn.ReLU()),
                    ("first_pool", nn.MaxPool2d(2)),
                    ("second_conv", nn.Conv2d(6, 16, 3)),
                    ("second_relu", nn.ReLU()),
                    ("second_pool", nn.MaxPool2d(2)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("first_linear", nn.Linear(16 * 5 * 5, 120)),
                    ("first_relu", nn.ReLU()),
                    ("second_linear", nn.Linear(120, 84)),
                    ("second_relu", nn.ReLU()),
                    ("third_linear", nn.Linear(84, 10)),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
