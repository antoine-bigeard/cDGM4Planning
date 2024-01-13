import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 1024))
        z = self.sampling(mu, log_var)
        return self.decoder(z).view(-1, 1, 32, 32), mu, log_var


class ConvVAE(nn.Module):
    def __init__(self, z_dim, conditional=False):
        super(ConvVAE, self).__init__()

        self.conditional = conditional

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.enc_fc = nn.Linear(4 * 4 * 128, 4 * 4 * 64)
        self.enc_fc1 = nn.Linear(4 * 4 * 128, z_dim)
        self.enc_fc2 = nn.Linear(4 * 4 * 128, z_dim)

        # Decoder
        if conditional:
            self.enc_layer = nn.Linear(2 * 28 * 28, 512)
        self.dec_fc1 = (
            nn.Linear(z_dim, 4 * 4 * 128)
            if not conditional
            else nn.Linear(z_dim + 512, 4 * 4 * 128)
        )
        # self.dec_fc2 = nn.Linear(4 * 4 * 64, 4 * 4 * 128)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=1)

    def encoder(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        # h = F.relu(self.enc_fc(h))
        return self.enc_fc1(h), self.enc_fc2(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z, c=None):
        if self.conditional:
            c = self.enc_layer(c.view(-1, 28 * 28))
            z = torch.cat((z, c), dim=1)
        h = F.relu(self.dec_fc1(z))
        # h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 128, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 32 * 32)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 32, 32))


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(32 * 32, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


"""
class Encoder(nn.Module):
    def __init__(self, nz):
        super(Encoder, self).__init__()
        self.nz = nz
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.mean = nn.Linear(8 * 8 * 64, nz)
        self.log_var = nn.Linear(8 * 8 * 64, nz)

    def forward(self, x):
        x = x.float().unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, nz, nd):
        super(Decoder, self).__init__()
        self.nz = nz
        self.nd = nd
        self.fc = nn.Linear(nz + nd, 8 * 8 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.float()
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x
"""
