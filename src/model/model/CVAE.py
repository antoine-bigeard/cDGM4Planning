import torch
import torch.nn as nn

from src.utils import idx2onehot


class VAE(nn.Module):
    def __init__(
        self,
        encoder_layer_sizes,
        latent_dim,
        decoder_layer_sizes,
        conditional=False,
    ):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_dim) == int
        assert type(decoder_layer_sizes) == list

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim, conditional)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim, conditional)

    def forward(self, x, c=None):

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim, conditional):

        super().__init__()

        self.conditional = conditional

        self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64, layer_sizes[0]))

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, c=None):

        x = torch.cat((x, c), dim=1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim, conditional):

        super().__init__()

        self.latent_layer = nn.Linear(latent_dim, 64)
        self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64, layer_sizes[0]))

        self.conditional = conditional
        input_size = layer_sizes[0]

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        z = self.latent_layer(z.cuda()).unsqueeze(1)
        z = torch.cat((z, c.cuda()), dim=1)

        x = self.MLP(z)

        return x
