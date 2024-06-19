from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier


class BinaryClassifier(nn.Module):
    def __init__(self, input_params: int = 6):
        super().__init__()
        self.layer1 = nn.Linear(input_params, 60)
        self.layer2 = nn.Linear(60, 60)
        self.layer3 = nn.Linear(60, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x


def get_skorch_model(input_params: int = 6):
    model = NeuralNetBinaryClassifier(
        partial(BinaryClassifier, input_params=input_params),
        optimizer=optim.AdamW,
        criterion=nn.BCELoss,
        max_epochs=10,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return model
