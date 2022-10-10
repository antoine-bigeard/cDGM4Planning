import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, sampling="upsample", kernel_size=3, stride=1, padding=1
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sampling = sampling

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                self.in_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.out_ch, 0.8),
            nn.ReLU(),
        )
        if self.sampling == "upsample":
            self.up_x = nn.Upsample(scale_factor=2)
            self.conv_x = nn.Conv1d(
                self.in_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self.conv1.append(nn.Upsample(scale_factor=2))
        if self.sampling == "downsample":
            self.down_x = nn.AvgPool1d(2)
            self.conv_x = nn.Conv1d(
                self.in_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self.conv1.append(nn.AvgPool1d(2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                self.out_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.out_ch, 0.8),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        if self.sampling == "upsample":
            return out2 + self.up_x((self.conv_x(x)))
        if self.sampling == "downsample":
            return out2 + self.down_x((self.conv_x(x)))
        else:
            return out2 + x


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
