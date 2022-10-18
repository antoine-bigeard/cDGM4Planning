import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import idx2onehot


# class Block(nn.Module):
#     def __init__(
#         self,
#         in_ch,
#         middle_ch,
#         out_ch,
#         down_rate=None,
#         residual=False,
#     ):
#         super().__init__()
#         self.down_rate = down_rate
#         self.residual = residual
#         self.c1 = nn.Conv1d(in_ch, middle_ch, 1)
#         self.c2 = nn.Conv1d(middle_ch, middle_ch, kernel_size=3, padding=1)
#         self.c3 = nn.Conv1d(middle_ch, middle_ch, kernel_size=3, padding=1)
#         self.c4 = nn.Conv1d(middle_ch, out_ch, kernel_size=1)

#     def forward(self, x):
#         xhat = self.c1(F.gelu(x))
#         xhat = self.c2(F.gelu(xhat))
#         xhat = self.c3(F.gelu(xhat))
#         xhat = self.c4(F.gelu(xhat))
#         out = x + xhat if self.residual else xhat
#         if self.down_rate is not None:
#             out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
#         return out


# class Encoder(nn.Module):
#     def __init__(self, blocks=[()]):
#         self.in_conv = nn.Conv1d(3, 64)
#         enc_blocks = []
#         for res, down_rate in blockstr:
#             enc_blocks.append(
#                 Block(
#                     self.widths[res],
#                     int(self.widths[res] * H.bottleneck_multiple),
#                     self.widths[res],
#                     down_rate=down_rate,
#                     residual=True,
#                     use_3x3=use_3x3,
#                 )
#             )
#         n_blocks = len(blockstr)
#         for b in enc_blocks:
#             b.c4.weight.data *= np.sqrt(1 / n_blocks)
#         self.enc_blocks = nn.ModuleList(enc_blocks)

#     def forward(self, x):
#         x = x[
#             ..., -self.H.image_size :, :
#         ]  # does nothing unless it is edges2shoes or similar
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.in_conv(x)
#         activations = {}
#         activations[x.shape[2]] = x
#         for block in self.enc_blocks:
#             x = block(x)
#             res = x.shape[2]
#             x = (
#                 x
#                 if x.shape[1] == self.widths[res]
#                 else pad_channels(x, self.widths[res])
#             )
#             activations[res] = x
#         return activations


class Encoder(nn.Module):
    def __init__(self, latent_dim):

        super().__init__()
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.Conv1d(3, 64, 1, padding=0),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(8192, self.latent_dim),
        )

    def forward(self, x, c=None):

        x = torch.cat((x, c), dim=1)

        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):

        super().__init__()

        self.latent = nn.Linear(latent_dim, 8192)

        self.convs = nn.Sequential(
            nn.Conv1d(130, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z, c):
        z = self.latent(z.cuda()).view([z.size(0), 128, 64])
        input_dec = torch.cat((z, c.cuda()), dim=1)

        return self.convs(input_dec)


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        encoder_model: nn.Module = Encoder,
        decoder_model: nn.Module = Decoder,
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = encoder_model(latent_dim)
        self.decoder = decoder_model(latent_dim)

    def forward(self, x, c=None):

        z = self.encoder(x, c)
        recon_x = self.decoder(z, c)

        return recon_x, z

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x
