class SmallGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8192),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25, inplace=False),
            nn.Conv1d(
                in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                stride=2,
                kernel_size=4,
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                stride=2,
                kernel_size=4,
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 64])
        input_gen = torch.cat([z, y], dim=1)
        return self.common(input_gen)


class SmallDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.4, inplace=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.4, inplace=False),
            Flatten(),
            nn.Linear(2048, 1),
        )

    def forward(self, x, y):
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)


class InjectionGenerator(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 496),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.up_conv_block0 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            UpConvBlock(
                128,
                124,
                scale_factor=2,
                kernel_size=3,
                stride=1,
                padding=1,
                mode="linear",
            ),
            nn.BatchNorm1d(124),
            nn.LeakyReLU(0.2),
        )
        self.up_conv_block1 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            UpConvBlock(
                256,
                248,
                scale_factor=2,
                kernel_size=3,
                stride=1,
                padding=1,
                mode="linear",
            ),
            nn.BatchNorm1d(248),
            nn.LeakyReLU(0.2),
        )
        self.up_conv_block2 = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            UpConvBlock(
                512,
                496,
                scale_factor=2,
                kernel_size=3,
                stride=1,
                padding=1,
                mode="linear",
            ),
            nn.BatchNorm1d(496),
            nn.LeakyReLU(0.2),
        )
        self.up_conv_block3 = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            UpConvBlock(
                256,
                248,
                scale_factor=2,
                kernel_size=3,
                stride=1,
                padding=1,
                mode="linear",
            ),
            nn.BatchNorm1d(248),
            nn.LeakyReLU(0.2),
        )

        self.up_conv_block4 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 62, 3, padding=1),
            nn.BatchNorm1d(62),
            nn.LeakyReLU(0.2),
        )

        self.last_conv = nn.Conv1d(64, 1, 3, 1, padding=1)

        self.label_conv0 = nn.Conv1d(2, 4, kernel_size=1, stride=2**4)
        self.label_conv1 = nn.Conv1d(2, 4, kernel_size=1, stride=2**3)
        self.label_conv2 = nn.Conv1d(2, 8, kernel_size=1, stride=2**2)
        self.label_conv3 = nn.Conv1d(2, 16, kernel_size=1, stride=2**1)
        self.label_conv4 = nn.Conv1d(2, 8, kernel_size=1, stride=1)
        self.label_conv5 = nn.Conv1d(2, 2, kernel_size=1, stride=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        if y.size(2) == 51:
            y = self.upsample_label(y)

        out = self.latent(z).view([z.size(0), 124, 4])

        for i in range(5):
            ds_y = eval(f"self.label_conv{i}", {"self": self})(y)
            inp = torch.cat([out, ds_y], dim=1)
            out = eval(f"self.up_conv_block{i}", {"self": self})(inp)
        ds_y = eval(f"self.label_conv5", {"self": self})(y)
        inp = torch.cat([out, ds_y], dim=1)
        out = self.last_conv(inp)

        return out


class InjectionDiscriminator(nn.Module):
    def __init__(
        self,
        out_chs=[64, 128, 256, 512],
    ):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv1d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            # Flatten(),
            # nn.Linear(2048, 1)
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(
            #     in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            # ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(2048, 1),
        )

    def forward(self, x, y):
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)


class InjectionDiscriminator2(nn.Module):
    def __init__(self, out_chs=[64, 128, 256, 512]):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(2048, 1),
        )

    def forward(self, x, y):
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)


class AtienzaGenerator(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.conv_y = nn.Conv1d(32, 64, 3, stride=1, padding=1)

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 3840),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            ResidualBlock(1024, 512, sampling="upsample"),
            ResidualBlock(512, 256, sampling="upsample"),
            ResidualBlock(256, 128, sampling="upsample"),
            ResidualBlock(128, 64, sampling="upsample"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        y = y.view(y.size(0), 32, 4)
        y = self.conv_y(y)
        z = self.latent(z).view([z.size(0), 960, 4])
        inp = torch.cat([z, y], dim=1)

        return self.common(inp)


class AtienzaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.common = nn.Sequential(
            ResidualBlock(3, 64, sampling="downsample"),
            ResidualBlock(64, 128, sampling="downsample"),
            ResidualBlock(128, 256, sampling="downsample"),
            # ResidualBlock(256, 512, sampling="downsample"),
            nn.AvgPool1d(4, stride=2, padding=1),
            Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        input_disc = torch.cat([x, y], dim=1)

        return self.common(input_disc)


class CondAtienzaRes(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8064),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            ResidualBlock(128, 256, sampling=None),
            ResidualBlock(256, 512, sampling=None),
            ResidualBlock(512, 256, sampling=None),
            ResidualBlock(256, 128, sampling=None),
            ResidualBlock(128, 64, sampling=None),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        # self.conv_y = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # adapted_y = self.conv_y(y)
        input = self.latent(z).view([z.size(0), 126, 64])
        input = torch.cat([input, y], dim=1)
        return self.common(input)


class CondAtienzaResDownUp(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8064),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            ResidualBlock(128, 256, sampling="downsample"),
            ResidualBlock(256, 512, sampling="downsample"),
            ResidualBlock(512, 256, sampling="upsample"),
            ResidualBlock(256, 128, sampling="upsample"),
            ResidualBlock(128, 64, sampling=None),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        # self.conv_y = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # adapted_y = self.conv_y(y)
        input = self.latent(z).view([z.size(0), 126, 64])
        input = torch.cat([input, y], dim=1)
        return self.common(input)


class CondAtienzaResDownUpInject(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8064),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common1 = nn.Sequential(
            ResidualBlock(128, 256, sampling="downsample"),
            ResidualBlock(256, 512, sampling="downsample"),
            ResidualBlock(512, 256, sampling="upsample"),
            ResidualBlock(256, 126, sampling="upsample"),
        )
        self.common2 = nn.Sequential(
            ResidualBlock(128, 64, sampling=None),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        # self.conv_y = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # adapted_y = self.conv_y(y)
        input = self.latent(z).view([z.size(0), 126, 64])
        input = torch.cat([input, y], dim=1)
        out = self.common1(input)
        out = torch.cat([out, y], dim=1)
        return self.common2(out)


class CondAtienzaNoRes(nn.Module):
    def __init__(self, latent_dim, n_up_conv_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_up_conv_blocks = n_up_conv_blocks

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8064),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        # self.conv_y = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # adapted_y = self.conv_y(y)
        input = self.latent(z).view([z.size(0), 126, 64])
        input = torch.cat([input, y], dim=1)
        return self.common(input)


class LargerGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8192),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25, inplace=False),
            nn.Conv1d(
                in_channels=256, out_channels=512, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=128, stride=2, kernel_size=4, padding=1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=1, stride=2, kernel_size=4, padding=1
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 64])
        input_gen = torch.cat([z, y], dim=1)
        return self.common(input_gen)


class LargerGenerator2(nn.Module):
    def __init__(
        self,
        latent_dim,
        spectral_norm_layers: list = [False, False, False, False],
        spec_norm_lin=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spectral_norm_layers = spectral_norm_layers
        self.spec_norm_lin = spec_norm_lin

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8192),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25, inplace=False),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=512, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Conv1d(
            #     in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1
            # ),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 64])
        input_gen = torch.cat([z, y], dim=1)
        return self.common(input_gen)


class LargerGenerator3(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 8192),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25, inplace=False),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=512, stride=2, kernel_size=4, padding=1
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=512, out_channels=256, stride=2, kernel_size=4, padding=1
            ),
            nn.LeakyReLU(),
        )
        self.common2 = nn.Sequential(
            nn.Conv1d(
                in_channels=258, out_channels=128, stride=1, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=1, stride=1, kernel_size=3, padding=1
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 64])
        input_gen = torch.cat([z, y], dim=1)
        out = self.common(input_gen)
        return self.common2(torch.cat([out, y], dim=1))
