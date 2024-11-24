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
        # CNN encoding
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

    def forward(self, x):
        encoded_outputs, hidden = self.encoder(x)
        reconstructed_outputs = self.decoder(encoded_outputs)
        return reconstructed_outputs


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=[19, 39, 59]):
        super(InceptionModule, self).__init__()
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
            # InceptionModule outputs num_filters * 4 channels
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

# ... (your existing inception encoder code)

# Define the Decoder
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
            # Calculate desired length after this layer
            desired_length = self.input_length // (2 ** (depth - d - 1))
            # Calculate current output length without output_padding
            output_length = (current_length - 1) * stride - 0 + kernel_size
            output_padding = desired_length - output_length
            # Ensure output_padding is non-negative and less than stride
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