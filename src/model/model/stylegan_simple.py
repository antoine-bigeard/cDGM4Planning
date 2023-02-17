"""
This code for StyleGan is adapted from `https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/gan/stylegan/__init__.py`. 
"""

import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from src.model.model.blocks import Flatten


class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        return self.net(z)


class Generator(nn.Module):
    """
    StyleGAN2 Generator
    The generator starts with a learned constant.
    Then it has a series of blocks. The feature map resolution is doubled at each block
    Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.
    """

    def __init__(
        self,
        log_resolution: int,
        d_latent: int,
        n_features: int = 32,
        max_features: int = 512,
        n_gen_blocks: int = 8,
        add_noise: bool = True,
        smooth=True,
        **kwargs,
    ):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `d_latent` is the dimensionality of $w$
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        self.add_noise = add_noise
        self.smooth = smooth
        self.n_features = n_features
        # Something like `[512, 512, 256, 128, 64, 32]`
        features = [
            min(max_features, n_features * (2**i))
            for i in range(log_resolution - 2, -1, -1)
        ]
        self.features = features
        # Number of generator blocks
        self.n_blocks = len(features)

        self.mapping_network = MappingNetwork(features=n_features, n_layers=8)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        # Generator blocks
        blocks = [
            GeneratorBlock(d_latent, features[i - 1], features[i])
            for i in range(1, self.n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.up_sample = UpSample(smooth=self.smooth)

        self.n_gen_blocks = n_gen_blocks

        self.encode_cond = nn.Sequential(Flatten(), nn.Linear(2 * 32 * 32, 32))

    def get_noise(self, batch_size: int):
        """
        ### Generate noise
        This generates noise for each [generator block](index.html#generator_block)
        """
        noise = []
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution).cuda()
            n2 = torch.randn(batch_size, 1, resolution, resolution).cuda()

            noise.append((n1, n2))

            resolution *= 2

        # Return noise tensors
        return noise

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        # input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    ):
        """
        * `w` is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate
        $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]`.
        * `input_noise` is the noise for each block.
        It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs
        after each convolution layer (see the diagram).
        """

        # Get batch size
        y = self.encode_cond(y)
        z = torch.cat(
            [z.squeeze().unsqueeze(0) if z.size(0) == 1 else z.squeeze(), y], axis=1
        )
        w = self.mapping_network(
            z.squeeze().unsqueeze(0) if z.size(0) == 1 else z.squeeze()
        )
        w = w[None, :, :].expand(self.n_gen_blocks, -1, -1)
        batch_size = w.shape[1]
        input_noise = self.get_noise(batch_size)
        if not self.add_noise:
            input_noise = [(None, None) for i in range(len(input_noise))]

        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # The first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Get first rgb image
        rgb = self.to_rgb(x, w[0])

        # Evaluate rest of the blocks
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new

        # Return the final RGB image
        return rgb

    def inference(self, y: torch.Tensor, latent_dim=20):
        with torch.inference_mode():
            z = torch.randn(y.shape[0], 1, latent_dim).cuda()
            return self.forward(z, y)


class GeneratorBlock(nn.Module):
    """
    ### Generator Block
    The generator block consists of two [style blocks](#style_block) ($3 \times 3$ convolutions with style modulation)
    and an RGB output.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        # Second [style block](#style_block)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        # *toRGB* layer
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])

        # Get RGB image
        rgb = self.to_rgb(x, w)

        # Return feature map and rgb image
        return x, rgb
        # return x


class StyleBlock(nn.Module):
    """
    ### Style Block
    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):

        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    ### To RGB
    In our case ToRGB generates only 1 channel in output.
    """

    def __init__(self, d_latent: int, features: int):

        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector s
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation

    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        demodulate: float = True,
        eps: float = 1e-8,
    ):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt(
                (weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    """
    ## StyleGAN 2 Discriminator
    Discriminator first transforms the image to a feature map of the same resolution and then
    runs it through a series of blocks with residual connections.
    The resolution is down-sampled by $2 \times$ at each block while doubling the
    number of features.
    """

    def __init__(
        self,
        log_resolution: int,
        n_features: int = 64,
        max_features: int = 512,
        smooth=True,
        **kwargs,
    ):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Layer to convert RGB image to a feature map with `n_features` number of features.
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [
            min(max_features, n_features * (2**i)) for i in range(log_resolution - 1)
        ]
        n_blocks = len(features) - 1
        # Discriminator blocks
        blocks = [
            DiscriminatorBlock(features[i], features[i + 1], smooth=smooth)
            for i in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        # [Mini-batch Standard Deviation](#mini_batch_std_dev)
        self.std_dev = MiniBatchStdDev()
        # Number of features after adding the standard deviations map
        final_features = features[-1]
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = torch.cat([x, y], axis=1)
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    """
    ### Discriminator Block
    Discriminator block consists of two 3x3 convolutions with a residual connection.
    """

    def __init__(self, in_features, out_features, smooth=True):
        super().__init__()
        self.residual = nn.Sequential(
            DownSample(smooth=smooth),
            EqualizedConv2d(in_features, out_features, kernel_size=1),
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample(smooth=smooth)

        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """
    ### Mini-batch Standard Deviation
    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)

        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    """
    ### Down-sample
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://papers.labml.ai/paper/1904.11486).
    """

    def __init__(self, smooth=True):
        super().__init__()
        self.smooth = smooth
        if self.smooth:
            self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        if self.smooth:
            x = self.smooth(x)
        return F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False
        )


class UpSample(nn.Module):
    """
    ### Up-sample
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://papers.labml.ai/paper/1904.11486).
    """

    def __init__(self, smooth=True):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # Smoothing layer
        self.smooth = smooth
        if self.smooth:
            self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        if self.smooth:
            return self.smooth(self.up_sample(x))
        return self.up_sample(x)


class Smooth(nn.Module):
    """
    ### Smoothing Layer
    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    ## Learning-rate Equalized Linear Layer
    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    ## Learning-rate Equalized 2D Convolution Layer
    This uses [learning-rate equalized weights](#equalized_weights) for a convolution layer.
    """

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int, padding: int = 0
    ):
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    ## Learning-rate Equalized Weights Parameter
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        # Multiply the weights by c and return
        return self.weight * self.c


class GradientPenalty(nn.Module):
    """
    ## Gradient Penalty
    """

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        * `x` is $x \sim \mathcal{D}$
        * `d` is $D(x)$
        """

        # Get batch size
        batch_size = x.shape[0]

        gradients, *_ = torch.autograd.grad(
            outputs=d, inputs=x, grad_outputs=d.new_ones(d.shape), create_graph=True
        )

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm**2)


class PathLengthPenalty(nn.Module):
    """
    ## Path Length Penalty
    """

    def __init__(self, beta: float):
        super().__init__()

        self.beta = beta

        self.steps = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True,
        )

        norm = (gradients**2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta**self.steps)
            # Calculate the penalty
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate a
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.0)

        return loss
