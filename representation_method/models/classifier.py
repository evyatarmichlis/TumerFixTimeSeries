import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.projection is not None:
            residual = self.projection(residual)
        out += residual
        out = self.relu(out)
        return out


class ComplexCNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ComplexCNNClassifier, self).__init__()
        self.initial_conv = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        self.layer1 = self._make_layer(ResidualBlock, 64, 64, 2, 1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 2, 2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 2, 4)
        self.layer4 = self._make_layer(ResidualBlock, 256, 512, 2, 8)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, dilation):
        layers = []
        layers.append(block(in_channels, out_channels, dilation=dilation))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.dropout(x)

        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)

        x = self.fc(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(CombinedModel, self).__init__()
        self.encoder = encoder.encoder
        self.classifier = ComplexCNNClassifier(input_dim=128, num_classes=num_classes)

    def forward(self, x):
        outputs, hidden = self.encoder(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        x = outputs.permute(0, 2, 1)
        logits = self.classifier(x)
        return logits