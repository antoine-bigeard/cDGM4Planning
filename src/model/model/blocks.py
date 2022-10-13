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
            nn.BatchNorm1d(self.out_ch),
            nn.ReLU(),
        )
        if self.sampling == "upsample":
            self.up_x = nn.Upsample(scale_factor=2)
            self.conv1.append(nn.Upsample(scale_factor=2))
            # self.up_x = nn.ConvTranspose1d(self.out_ch, self.out_ch, 4, 2, padding=1)
            # self.conv1.append(
            #     nn.ConvTranspose1d(self.out_ch, self.out_ch, 4, 2, padding=1)
            # )
        if self.sampling == "downsample":
            self.down_x = nn.AvgPool1d(2)
            self.conv1.append(nn.AvgPool1d(2))
            # self.down_x = nn.Conv1d(self.out_ch, self.out_ch, 4, stride=2, padding=1)
            # self.conv1.append(
            #     nn.Conv1d(self.out_ch, self.out_ch, 4, stride=2, padding=1)
            # )
        if self.out_ch != self.in_ch:
            self.conv_x = nn.Conv1d(
                self.in_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                self.out_ch,
                self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        if self.in_ch != self.out_ch:
            inp_x = self.conv_x(x)
        if self.sampling == "upsample":
            return out2 + self.up_x(inp_x)
        if self.sampling == "downsample":
            return out2 + self.down_x(inp_x)
        else:
            return out2 + inp_x


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


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, d_conv=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3), nn.ReLU(), nn.Conv1d(out_ch, out_ch, 3)
        )

    def forward(self, x):
        return self.net(x)
