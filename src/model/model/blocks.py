import torch.nn as nn


import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from src.utils import *


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def clean_padding_y(y, pad_value=-1):
    no_padding = torch.where(y[:, 0, 0] > -1)[0]
    return y[no_padding, :, :]


class SequentialCat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoding_layer = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x):
        return self.encoding_layer(x).squeeze()


class ConditionEncoding(nn.Module):
    def __init__(
        self, input_size=128, hidden_size=128, num_layers=1, batch_first=True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.recurrent_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, y):
        out = torch.zeros(y.shape[0], 2, self.hidden_size // 2)
        for i, seq in enumerate(y):
            unpadded_seq = clean_padding_y(seq)
            unpadded_seq = unpadded_seq.reshape(
                unpadded_seq.shape[0], unpadded_seq.shape[1] * unpadded_seq.shape[2]
            )
            out[i] = self.recurrent_layer(unpadded_seq)[0][-1].reshape(
                2, self.hidden_size // 2
            )
        return out


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


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


LAYERS = {
    "Cv": nn.Conv1d,
    "CvT": nn.ConvTranspose1d,
    "BN": nn.BatchNorm1d,
    "Cv2d": nn.Conv2d,
    "CvT2d": nn.ConvTranspose2d,
    "BN2d": nn.BatchNorm2d,
    "LR": nn.LeakyReLU,
    "D": nn.Dropout,
    "L": nn.Linear,
    "F": Flatten,
}


# class BasicBlock(nn.Module):
#     def __init__(self, layer_params, injection=None) -> None:
#         super().__init__()
#         layers = []
#         for key, item in layer_params.items():
#             if injection is not None and injection[0] and key in ["Cv", "CvT"]:
#                 cv_param = list(item)
#                 cv_param[0] += test0(injection[1], 2)
#                 layers.append(
#                     LAYERS[key](*cv_param) if cv_param is not None else LAYERS[key]()
#                 )
#             else:
#                 layers.append(LAYERS[key](*item) if item is not None else LAYERS[key]())
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layers(x)


def BasicBlock(layer_params, injection=None):
    layers = []
    for key, item in layer_params.items():
        if (
            injection is not None
            and injection[0]
            and key in ["Cv", "CvT", "Cv2d", "CvT2d"]
        ):
            cv_param = list(item)
            cv_param[0] += test0(injection[1], 2)
            layers.append(
                LAYERS[key](*cv_param) if cv_param is not None else LAYERS[key]()
            )
        else:
            layers.append(LAYERS[key](*item) if item is not None else LAYERS[key]())
    if len(layers) > 1:
        return nn.Sequential(*layers)
    return layers[0]


def BasicBlockY(injection):
    injection = injection
    if not injection[0] or (injection[1] == 0 and injection[2] == 1):
        trans = nn.Identity()
    elif injection[1] == 0:
        trans = (
            nn.Upsample(scale_factor=injection[2])
            if injection[3] > 1
            else nn.AvgPool1d(kernel_size=4, stride=1 / injection[2], padding=1)
        )
    elif injection[1] > 0 and injection[2] == 1:
        trans = nn.Conv1d(2, injection[1], 3, 1, 1)
    else:
        if injection[2] < 1:
            trans = nn.Conv1d(2, injection[1], 4, int(1 / injection[2]), 1)
        else:
            trans = nn.ConvTranspose1d(2, injection[1], 4, injection[2], 1)
    return trans


def BasicBlockY2d(injection):
    injection = injection
    if not injection[0] or (injection[1] == 0 and injection[2] == 1):
        trans = nn.Identity()
    elif injection[1] == 0:
        trans = (
            nn.Upsample(scale_factor=injection[2])
            if injection[3] > 1
            else nn.AvgPool2d(kernel_size=4, stride=1 / injection[2], padding=1)
        )
    elif injection[1] > 0 and injection[2] == 1:
        trans = nn.Conv2d(2, injection[1], 3, 1, 1)
    else:
        if injection[2] < 1:
            trans = nn.Conv2d(2, injection[1], 4, int(1 / injection[2]), 1)
        else:
            trans = nn.ConvTranspose2d(2, injection[1], 4, injection[2], 1)
    return trans


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# class BasicBlockY(nn.Module):
#     def __init__(self, injection) -> None:
#         super().__init__()
#         self.injection = injection
#         if not self.injection[0] or (self.injection[1] == 0 and self.injection[2] == 1):
#             self.trans = nn.Identity()
#         elif self.injection[1] == 0:
#             self.trans = (
#                 nn.Upsample(scale_factor=self.injection[2])
#                 if self.injection[3] > 1
#                 else nn.AvgPool1d(
#                     kernel_size=4, stride=1 / self.injection[2], padding=1
#                 )
#             )
#         elif self.injection[1] > 0 and self.injection[2] == 1:
#             self.trans = nn.Conv1d(2, self.injection[1], 3, 1, 1)
#         else:
#             if self.injection[2] > 1:
#                 self.trans = nn.Conv1d(2, self.injection[1], 4, self.injection[2], 1)
#             else:
#                 self.trans = nn.ConvTranspose1d(
#                     2, self.injection[1], 4, int(1 / self.injection[2]), 1
#                 )

#     def forward(self, x):
#         return self.trans(x)
