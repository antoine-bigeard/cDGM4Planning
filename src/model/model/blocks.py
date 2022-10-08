import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.in_ch = in_ch

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_ch, 0.8),
            nn.ReLU(),
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_ch, 0.8),
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConvBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, scale_factor, kernel_size, stride, padding, mode="linear"
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                self.in_ch, self.out_ch, self.kernel_size, self.stride, self.padding
            ),
            # nn.ConvTranspose1d(
            #     self.out_ch,
            #     self.out_ch,
            #     self.kernel_size + 1,
            #     self.scale_factor,
            #     self.padding,
            # )
            nn.Upsample(scale_factor=self.scale_factor, mode=self.mode),
        )

    def forward(self, x):
        return self.conv_block(x)
