import torch


class ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x


class ViTEncoder(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = [ViTEncoderLayer(num_layers) for _ in range(num_layers)]

    def forward(self, x):
        return x


class ViTEncoderLayer(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers: list[torch.nn.Module] = []

    def forward(self, x):
        return x


class ResidualLayer(torch.nn.Module):
    def __init__(self, init_layer: torch.nn.Module):
        super().__init__()
        self.init_layer = init_layer

    def forward(self, x):
        return x
