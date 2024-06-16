import torch.nn as nn

#Submodele do łatwiejszego tworzenia sieci
class FCBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_p) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dropout(self.fc(x)))
    

class ConvolutionBlock(nn.Module):
    def __init__(self, filter_in, filter_out, conv_size, conv_stride, activation=True, pooling=False, batch_norm=True) -> None:
        super().__init__()
        self.layers = [
            nn.Conv2d(filter_in, filter_out, conv_size, conv_stride, padding=conv_size//2),
        ]
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(filter_out))
        if activation:
            self.layers.append(nn.ReLU())
        if pooling:
            self.layers.append(nn.MaxPool2d(2))

        self.structure = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.structure(x)


class ResidualBlock(nn.Module):
    def __init__(self, filter_in, filter_out, conv_size, conv_stride, residual_conv=False, pooling=False) -> None:
        super().__init__()
        self.layers = [
            ConvolutionBlock(filter_in, filter_out, conv_size, conv_stride),
            ConvolutionBlock(filter_out, filter_out, conv_size, conv_stride, activation=False, pooling=pooling),
        ]
        self.non_residual_path = nn.Sequential(*self.layers)
        self.residual_path = nn.Conv2d(filter_in, filter_out, 1, 1) if residual_conv else None
        self.residual_pooling = nn.MaxPool2d(2) if pooling else None
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.residual_path(x) if self.residual_path else x
        residual = self.residual_pooling(residual) if self.residual_pooling else residual
        non_residual = self.non_residual_path(x)
        return self.activation(non_residual + residual)