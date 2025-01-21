import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNRecurrentEncoder(nn.Module):

    def __init__(self, in_channels, num_filters, depth, hidden_size, num_layers=1, rnn_type='GRU'):

        super(CNNRecurrentEncoder, self).__init__()
        self.conv_encoder = InceptionEncoder(in_channels, num_filters, depth)
        self.encoder_channels = self.conv_encoder.get_channels()
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.rnn_type = rnn_type

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.encoder_channels[-1],
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)

        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.encoder_channels[-1],
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.dropout1 = nn.Dropout(0.5)  # After CNN
        self.dropout2 = nn.Dropout(0.5)  # After RNN

        self.encoded_length = None

    def forward(self, x):
        x = self.conv_encoder(x)
        self.encoded_length = x.size(2)
        x = x.permute(0, 2, 1)
        outputs, hidden = self.rnn(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs, hidden

    def get_channels(self):
        return self.encoder_channels

    def get_encoded_length(self):
        return self.encoded_length


class CNNRecurrentDecoder(nn.Module):
    def __init__(self, num_filters, depth, encoder_channels, input_length, hidden_size, num_layers=1, rnn_type='GRU'):
        super(CNNRecurrentDecoder, self).__init__()
        self.hidden_size = hidden_size

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=encoder_channels[-1], num_layers=num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=encoder_channels[-1], num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'")

        self.conv_decoder = InceptionDecoder(num_filters=num_filters, depth=depth, encoder_channels=encoder_channels, input_length=input_length)

    def forward(self, x):
        outputs, hidden = self.rnn(x)

        x = outputs.permute(0, 2, 1)  # [batch_size, channels, seq_length]
        x = self.conv_decoder(x)
        return x

class CNNRecurrentAutoencoder(nn.Module):
    def __init__(self, in_channels, num_filters, depth, hidden_size, num_layers=1, rnn_type='GRU', input_length=None):
        super(CNNRecurrentAutoencoder, self).__init__()


        self.encoder = CNNRecurrentEncoder(
            in_channels=in_channels,
            num_filters=num_filters,
            depth=depth,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type
        )
        encoder_channels = self.encoder.get_channels()
        if input_length is None:
            input_length = self.encoder.get_encoded_length()
        self.decoder = CNNRecurrentDecoder(
            num_filters=num_filters,
            depth=depth,
            encoder_channels=encoder_channels,
            input_length=input_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        encoded_outputs, hidden = self.encoder(x)
        reconstructed_outputs = self.decoder(encoded_outputs)
        hidden_for_class = hidden[-1] if hidden.dim() == 3 else hidden
        logits = self.classifier(hidden_for_class)
        return reconstructed_outputs,logits


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32,
                 kernel_sizes=[5, 15, 35]):  # Match the old kernel sizes 9, 19, 39]
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.conv_layers.append(
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=padding)
            )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.bn = nn.BatchNorm1d((len(kernel_sizes) + 1) * out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        outputs = [conv(x_bottleneck) for conv in self.conv_layers]
        outputs.append(self.maxpool_conv(x))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class InceptionEncoder(nn.Module):
    def __init__(self, in_channels, num_filters, depth):
        super(InceptionEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.channels = [in_channels]  # Initialize with input channels
        current_channels = in_channels
        for d in range(depth):
            self.blocks.append(InceptionModule(current_channels, num_filters))
            current_channels = num_filters * 4
            self.channels.append(current_channels)
            self.blocks.append(nn.MaxPool1d(kernel_size=2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_channels(self):

        return self.channels



class InceptionDecoder(nn.Module):

    def __init__(self, num_filters, depth, encoder_channels, input_length):
        super(InceptionDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        self.input_length = input_length
        current_length = 1  # Start from the encoded length
        for d in range(depth):
            input_channels = decoder_channels[d]
            output_channels = decoder_channels[d + 1]
            stride = 2
            kernel_size = 2
            desired_length = self.input_length // (2 ** (depth - d - 1))
            output_length = (current_length - 1) * stride - 0 + kernel_size
            output_padding = desired_length - output_length
            if output_padding < 0 or output_padding >= stride:
                output_padding = 0
            current_length = desired_length
            self.blocks.append(
                nn.ConvTranspose1d(
                    input_channels,

                    output_channels,

                    kernel_size=kernel_size,

                    stride=stride,

                    output_padding=output_padding

                )

            )

            self.blocks.append(nn.ReLU())



    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        return x





# Define the Inception Autoencoder

class InceptionAutoencoder(nn.Module):

    def __init__(self, in_channels, input_length, num_filters=32, depth=3):

        super(InceptionAutoencoder, self).__init__()

        self.encoder = InceptionEncoder(in_channels, num_filters, depth)

        encoder_channels = self.encoder.get_channels()

        self.decoder = InceptionDecoder(num_filters, depth, encoder_channels, input_length)



    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded





def initialize_weights(m):

    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):

        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if m.bias is not None:

            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):

        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if m.bias is not None:

            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):

        nn.init.ones_(m.weight)

        nn.init.zeros_(m.bias)




class VAEEncoder(nn.Module):
    """Separate encoder module for compatibility with classifier"""

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder RNN
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Project to correct dimension for classifier compatibility
        self.projection = nn.Linear(hidden_dim * 2, 128)

    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [batch, features, time] -> [batch, time, features]
        x = x.transpose(1, 2)

        # Encode sequence
        outputs, hidden = self.encoder_rnn(x)  # [batch, time, hidden_dim * 2]

        # Project to 128 dimensions
        outputs = self.projection(outputs)  # [batch, time, 128]

        # Permute to [batch, 128, time]
        outputs = outputs.permute(0, 2, 1)  # Convert to [batch, 128, time]

        # Take single time step if needed
        if outputs.size(2) > 1:
            outputs = outputs[:, :, -1:]  # Keep last time step, shape: [batch, 128, 1]

        return outputs, hidden


class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()

        # Save dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Create encoder module that can be accessed directly
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        # Decoder components
        self.decoder_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # VAE specific layers
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

    def encode(self, x):
        """Get VAE encoding (mu, logvar)"""
        outputs, _ = self.encoder(x)
        # outputs shape is [batch, 128, 1]
        outputs = outputs.squeeze(-1)  # [batch, 128]

        # Get latent parameters
        mu = self.mu_layer(outputs)
        logvar = self.logvar_layer(outputs)

        return mu, logvar

    def decode(self, z, seq_len):
        # Initialize decoder state
        hidden = self.decoder_init(z).unsqueeze(0)

        # Create initial input
        batch_size = z.size(0)
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim).to(z.device)

        # Generate sequence
        output, _ = self.decoder_rnn(decoder_input, hidden)
        output = self.output_layer(output)

        # Return in same format as input [batch, features, time]
        return output.transpose(1, 2)

    def forward(self, x, return_embeddings=False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save original sequence length
        seq_len = x.size(2)  # Assuming input is [batch, features, time]

        # Get VAE encoding
        mu, logvar = self.encode(x)
        z = self.encoder.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z, seq_len)

        if return_embeddings:
            return recon, z
        return recon, mu, logvar

    def get_normalized_embeddings(self, x):
        """Get normalized embeddings for visualization or analysis"""
        mu, logvar = self.encode(x)
        z = self.encoder.reparameterize(mu, logvar)
        return F.normalize(z, p=2, dim=1)
# class TimeSeriesVAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
#         super().__init__()
#
#         # Save dimensions
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
#
#         # Encoder
#         self.encoder_rnn = nn.GRU(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True
#         )
#
#         # VAE components
#         self.mu_layer = nn.Linear(hidden_dim * 2, latent_dim)
#         self.logvar_layer = nn.Linear(hidden_dim * 2, latent_dim)
#
#         # Decoder
#         self.decoder_init = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_rnn = nn.GRU(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(hidden_dim, input_dim)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def encode(self, x):
#         # x shape: [batch, features, time] -> need [batch, time, features]
#         x = x.transpose(1, 2)
#
#         # Encode sequence
#         _, hidden = self.encoder_rnn(x)
#         # Concat bidirectional hidden states
#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#
#         # Get latent parameters
#         mu = self.mu_layer(hidden)
#         logvar = self.logvar_layer(hidden)
#
#         return mu, logvar
#
#     def decode(self, z, seq_len):
#         # Initialize decoder state
#         hidden = self.decoder_init(z).unsqueeze(0)
#
#         # Create initial input
#         batch_size = z.size(0)
#         decoder_input = torch.zeros(batch_size, seq_len, self.input_dim).to(z.device)
#
#         # Generate sequence
#         output, _ = self.decoder_rnn(decoder_input, hidden)
#         output = self.output_layer(output)
#
#         # Return in same format as input [batch, features, time]
#         return output.transpose(1, 2)
#
#     def forward(self, x):
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         # Save original sequence length
#         seq_len = x.size(2)  # Assuming input is [batch, features, time]
#
#         # Encode
#         mu, logvar = self.encode(x)
#
#         # Sample latent vector
#         z = self.reparameterize(mu, logvar)
#
#         # Decode
#         recon = self.decode(z, seq_len)
#
#         return recon, mu, logvar


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, weights


import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class ImprovedTimeSeriesVAE(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32, num_layers=2, nheads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 50, hidden_dim))

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # VAE components
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder components
        self.decoder_init = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.decoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        # x shape: [batch, features, time] -> [batch, time, features]
        x = x.transpose(1, 2)

        # Feature encoding
        x = self.feature_encoder(x)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer encoding
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        # Global pooling
        x = torch.mean(x, dim=1)  # [batch, hidden_dim]

        # Get latent parameters
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return mu, logvar

    def decode(self, z, seq_len):
        # Initialize decoder state
        hidden = self.decoder_init(z).unsqueeze(0)
        hidden = hidden.repeat(2, 1, 1)  # for bidirectional RNN

        # Create decoder input
        decoder_input = torch.zeros(z.size(0), seq_len, self.input_dim).to(z.device)

        # Run decoder
        output, _ = self.decoder_rnn(decoder_input, hidden)
        output = self.output_projection(output)

        # Return in original format [batch, features, time]
        return output.transpose(1, 2)

    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z, x.size(2))

        return recon, mu, logvar


import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        # Adjust convolutions to work with [batch, in_channels, hidden_dim]
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, dilation),
                               dilation=(1, dilation))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, dilation),
                               dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
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



class ResidualBlock_new(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
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


class CombinedVAEClassifier(nn.Module):
    """Classifier designed specifically to work with VAE encoder output"""

    def __init__(self, model, num_classes=2):
        super().__init__()
        self.encoder = model.encoder

        # Initial layers - work with [batch, 128, 1] input
        self.initial_conv = nn.Conv1d(128, 64, kernel_size=1)
        self.initial_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # Main layers
        self.layer1 = self._make_layer(ResidualBlock_new, 64, 64, 2, 1)
        self.layer2 = self._make_layer(ResidualBlock_new, 64, 128, 2, 2)
        self.layer3 = self._make_layer(ResidualBlock_new, 128, 256, 2, 4)

        # Global average pooling and final classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, dilation):
        layers = []
        layers.append(block(in_channels, out_channels, dilation=dilation))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Get encoded representation from VAE encoder
        encoded_output, _ = self.encoder(x)  # [batch, 128, 1]

        # Initial layers
        x = self.relu(self.initial_bn(self.initial_conv(encoded_output)))
        x = self.dropout(x)

        # Main layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Final classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x