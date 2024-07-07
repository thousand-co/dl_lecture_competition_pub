import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier2D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 1024
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock2D(in_channels, hid_dim),
            ConvBlock2D(int(hid_dim/2), int(hid_dim/2)),  # 7/7 hid_dim-->hid_dim/2
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(int(hid_dim/4), num_classes),  # 7/7 hid_dim-->hid_dim/4
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)

        self.dropout0 = nn.Dropout2d(p_drop)
        self.dropout1 = nn.Dropout2d(p_drop)
        self.dropout2 = nn.Dropout2d(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.dropout0(X)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.dropout1(X)

        X = self.conv2(X)
        X = F.glu(X, dim=-2)  # glu's output dim will be 1/2

        return self.dropout2(X)