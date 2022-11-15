import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim, hot_one_encoding: bool = True):
        super().__init__()
        self.hot_one_encoding = hot_one_encoding
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.Conv1d(3, 64, 1, padding=0),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        self.lin_mean = nn.Linear(8192, self.latent_dim)
        self.lin_var = nn.Linear(8192, self.latent_dim)

    def forward(self, x, y=None):

        x = torch.cat((x, y), dim=1)

        out = self.convs(x)

        mean = self.lin_mean(out)
        var = self.lin_var(out)

        return mean, var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hot_one_encoding: bool = True):
        super().__init__()
        self.hot_one_encoding = hot_one_encoding
        self.latent_dim = latent_dim

        self.latent = nn.Linear(
            latent_dim if self.hot_one_encoding else latent_dim + 2, 8192
        )

        self.convs = nn.Sequential(
            nn.Conv1d(
                130 if self.hot_one_encoding else 128,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z, y):
        if self.hot_one_encoding:
            z = self.latent(z.cuda()).view([z.size(0), 128, 64])
            input_dec = torch.cat((z, y.cuda()), dim=1)
        else:
            nonzero_idx = torch.where(y[:, 0, :])
            y = y[nonzero_idx[0], :, nonzero_idx[1]]
            y[:, 0] = nonzero_idx[1]
            z = torch.cat([z, y], dim=1)
            input_dec = self.latent(z.cuda()).view([z.size(0), 128, 64])

        return self.convs(input_dec)


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        encoder_model: nn.Module = Encoder,
        conf_encoder: dict = {},
        decoder_model: nn.Module = Decoder,
        conf_decoder: dict = {},
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.conf_encoder = conf_encoder
        self.conf_decoder = conf_decoder

        self.encoder = encoder_model(self.latent_dim, **self.conf_encoder)
        self.decoder = decoder_model(self.latent_dim, **self.conf_decoder)

    def forward(self, x, y):

        mean, var = self.encoder(x, y)
        z = self.reparameterize(mean, var)
        recon_x = self.decoder(z, y)

        return recon_x, mean, var, z

    def reparameterize(self, mean, var):

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)

        return mean + eps * std

    def inference(self, z, y):

        recon_x = self.decoder(z, y)

        return recon_x
