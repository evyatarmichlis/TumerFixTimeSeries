
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding.
    For 1D sequences, this is typically enough for time-series tasks.
    """
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices

        # Register pe as a buffer so it’s not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We add positional encoding up to seq_len positions.
        """
        seq_len = x.size(1)
        # Add the positional encoding to each batch’s slice
        x = x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Example time series classification model using a TransformerEncoder.
    """
    def __init__(self, input_dim, model_dim=64, n_heads=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, num_classes=2,
                 max_len=1000, batch_first=True):
        """
        Args:
            input_dim (int): Number of features per time step.
            model_dim (int): Embedding dimension for the Transformer.
            n_heads (int): Number of heads in multi-head attention.
            num_layers (int): Number of TransformerEncoder layers.
            dim_feedforward (int): Hidden dimension of the feed-forward network.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
            max_len (int): Max sequence length for positional encoding.
            batch_first (bool): Whether input is (batch_size, seq_len, embedding_dim).
        """
        super(TimeSeriesTransformer, self).__init__()

        self.model_dim = model_dim

        # 1) Project input_dim -> model_dim
        self.input_projection = nn.Linear(8, 64)
        # 2) Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len)

        # 3) Encoder layers
        #    Build a TransformerEncoder using nn.TransformerEncoderLayer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,  # So input is (batch, seq_len, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4) Classification head
        self.classifier = nn.Linear(model_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        # 1) Linear projection
        x = x.transpose(1, 2)
        x = self.input_projection(x)  # -> (batch_size, seq_len, model_dim)
        x = self.relu(x)

        # 2) Positional encoding
        x = self.pos_encoder(x)  # -> (batch_size, seq_len, model_dim)

        # 3) Transformer encoding
        #    (batch_size, seq_len, model_dim)
        x = self.transformer_encoder(x)  # -> same shape

        # 4) Pool over time dimension
        #    Example: global average pooling over seq_len
        x = x.mean(dim=1)  # -> (batch_size, model_dim)

        # 5) Classification
        x = self.dropout(x)
        x = self.classifier(x)  # -> (batch_size, num_classes)

        return x


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

        # In case of channel mismatch (in_channels != out_channels),
        # use 1x1 conv to match dimensions for the residual connection
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions differ, adjust the identity
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)
        return out


class ComplexCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_rnn_dim=256, rnn_type="GRU"):
        """
        Args:
            input_dim (int): Number of input channels (e.g. # features for 1D time series).
            hidden_rnn_dim (int): Hidden dimension for the GRU/LSTM layer.
            rnn_type (str): "GRU" or "LSTM".
        """
        super(ComplexCNNClassifier, self).__init__()
        self.initial_conv = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # Four CNN layers with increasing channels & dilation
        self.layer1 = self._make_layer(ResidualBlock, 64, 64, 4, dilation=1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 4, dilation=2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 4, dilation=4)
        self.layer4 = self._make_layer(ResidualBlock, 256, 512, 4, dilation=8)


        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=hidden_rnn_dim,
                num_layers=1,
                batch_first=True
            )
        else:  # Default to GRU
            self.rnn = nn.GRU(
                input_size=512,
                hidden_size=hidden_rnn_dim,
                num_layers=1,
                batch_first=True
            )


        self.fc = nn.Linear(hidden_rnn_dim, 2)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, dilation):
        layers = [block(in_channels, out_channels, dilation=dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, channels=input_dim, seq_len)
        """
        # CNN feature extraction
        x = self.relu(self.initial_bn(self.initial_conv(x)))  # (B, 64, seq_len)
        x = self.dropout(x)

        x = self.layer1(x)  # (B, 64, seq_len)
        x = self.dropout(x)

        x = self.layer2(x)  # (B, 128, seq_len)
        x = self.dropout(x)

        x = self.layer3(x)  # (B, 256, seq_len)
        x = self.dropout(x)

        x = self.layer4(x)  # (B, 512, seq_len)
        x = self.dropout(x)

        # Recurrent layer expects (B, seq_len, features)
        # So we transpose from (B, 512, seq_len) -> (B, seq_len, 512)
        x = x.transpose(1, 2)  # Now shape = (B, seq_len, 512)

        # Feed into GRU or LSTM
        if isinstance(self.rnn, nn.LSTM):
            # LSTM returns (output, (h_n, c_n))
            # We can take the last hidden state h_n (shape: (num_layers, B, hidden_rnn_dim))
            rnn_out, (h_n, c_n) = self.rnn(x)

            x = h_n[-1]  # last layer’s hidden state
        else:
            # GRU returns (output, h_n), where h_n is (num_layers, B, hidden_rnn_dim)
            rnn_out, h_n = self.rnn(x)
            x = h_n[-1]

        # x shape here = (B, hidden_rnn_dim)
        x = self.dropout(x)
        x = self.fc(x)  # (B, 2)
        return x


class CombinedModel(nn.Module):
    def __init__(self, model, num_classes):
        super(CombinedModel, self).__init__()
        self.encoder = model.encoder
        # self.classifier = ComplexCNNClassifier(input_dim=128, num_classes=num_classes)
        self.classifier =None

    def forward(self, x):

        x = x.view(x.size(0), -1)  # Flatten the input
        self.classifier = nn.Sequential(
            nn.Linear(x.size(1), 64),  # Dynamic input size
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        ).to(x.device)
        return self.classifier(x)


class SimpleMLP(nn.Module):
    """Simple MLP classifier head"""

    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        x = torch.mean(x, dim=2)  # Global average pooling over sequence
        return self.network(x)


class LinearClassifier(nn.Module):
    """Simple linear classifier with dropout"""

    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        x = torch.mean(x, dim=2)  # Global average pooling over sequence
        return self.network(x)


class SimpleConv1D(nn.Module):
    """Simple 1D CNN classifier"""

    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class SimpleCombinedModel(nn.Module):
    def __init__(self, model, num_classes, classifier_type='mlp'):
        super().__init__()
        self.encoder = model.encoder

        # Choose classifier based on type
        if classifier_type == 'mlp':
            self.classifier = SimpleMLP(input_dim=128, num_classes=num_classes)
        elif classifier_type == 'linear':
            self.classifier = LinearClassifier(input_dim=128, num_classes=num_classes)
        elif classifier_type == 'conv':
            self.classifier = SimpleConv1D(input_dim=128, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        outputs, hidden = self.encoder(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        x = outputs.permute(0, 2, 1)
        logits = self.classifier(x)
        return logits


class TVAEClassifier(nn.Module):
    def __init__(self, vae_model, hidden_dim=32, num_classes=2, dropout=0.3):
        super().__init__()
        self.vae_model = vae_model

        # Freeze VAE encoder initially
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # Get encoder output dimension from the VAE model
        self.encoder_dim = self.vae_model.latent_dim * 2  # Assuming bidirectional encoder

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, num_classes),
        )

        self.training_phase = 'frozen_encoder'  # or 'fine_tuning'

    def forward(self, x):
        # Get VAE encoder outputs
        mu, logvar = self.vae_model.encode(x)

        # Concatenate mean and logvar
        features = torch.cat((mu, logvar), dim=1)

        # Pass features through classifier
        logits = self.classifier(features)

        return logits

    def unfreeze_encoder(self):
        """Unfreeze VAE encoder for fine-tuning"""
        for param in self.vae_model.parameters():
            param.requires_grad = True
        self.training_phase = 'fine_tuning'


import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.scale = torch.sqrt(torch.FloatTensor([channels])).cuda()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = torch.matmul(Q.permute(0, 2, 1), K) / self.scale
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE attention
        self.se = SEBlock(out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)
        out += identity
        out = self.activation(out)
        return out





class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()

        # Initial convolution with larger kernel
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(128)

        # Enhanced residual blocks with increasing dilation
        self.layer1 = self._make_layer(128, 128, blocks=3, dilation=1)
        self.layer2 = self._make_layer(128, 256, blocks=3, dilation=2)
        self.layer3 = self._make_layer(256, 512, blocks=3, dilation=4)
        self.dropout =  nn.Dropout(0.5)
        # Multi-scale feature aggregation
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # Classifier head with class-balanced weights
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights with better scaling
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, dilation):
        layers = []
        layers.append(EnhancedResidualBlock(in_channels, out_channels, dilation))
        for _ in range(1, blocks):
            layers.append(EnhancedResidualBlock(out_channels, out_channels, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        # Apply temporal attention
        x = x + self.temporal_attention(x)

        # Process through residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)

        # Multi-scale feature aggregation
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        x = self.classifier(x)
        return x


class EnhancedCombinedModel(nn.Module):
    def __init__(self, model, num_classes=2):
        super().__init__()
        self.encoder = model.encoder
        self.classifier = EnhancedClassifier(input_dim=128, num_classes=num_classes)

    def forward(self, x):
        outputs, hidden = self.encoder(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        x = outputs.permute(0, 2, 1)

        logits = self.classifier(x)
        return logits