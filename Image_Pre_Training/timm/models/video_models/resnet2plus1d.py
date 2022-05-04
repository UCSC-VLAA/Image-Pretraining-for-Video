import einops
import torch
from .resnet3d import ResNet3d
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS, build_norm_layer, constant_init, kaiming_init
from torch.nn.modules.utils import _triple
import numpy as np

@CONV_LAYERS.register_module()
class Conv2plus1d(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.
    https://arxiv.org/pdf/1711.11248.pdf.
    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int | tuple[int]): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int | tuple[int]): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN3d')):
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False

        # The middle-plane is calculated according to:
        # M_i = \floor{\frac{t * d^2 N_i-1 * N_i}
        #   {d^2 * N_i-1 + t * N_i}}
        # where d, t are spatial and temporal kernel, and
        # N_i, N_i-1 are planes
        # and inplanes. https://arxiv.org/pdf/1711.11248.pdf
        mid_channels = 3 * (
            in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= (
            in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)


        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=bias)
        self.mid_channel = mid_channels
        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.conv_t(x)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)


@CONV_LAYERS.register_module()
class Conv2plus1d_reshape(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.
    https://arxiv.org/pdf/1711.11248.pdf.
    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int | tuple[int]): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int | tuple[int]): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN3d')):
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False

        # The middle-plane is calculated according to:
        # M_i = \floor{\frac{t * d^2 N_i-1 * N_i}
        #   {d^2 * N_i-1 + t * N_i}}
        # where d, t are spatial and temporal kernel, and
        # N_i, N_i-1 are planes
        # and inplanes. https://arxiv.org/pdf/1711.11248.pdf
        mid_channels = 3 * (
            in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= (
            in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)


        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=bias)
        self.mid_channel = mid_channels
        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.reshape_temporal_conv(self.conv_t, x)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)



    def reshape_temporal_conv(self, conv_3d, x, stride=1, groups=1):

        conv3d_para = conv_3d.weight
        outplanes, inplanes, _, _, _ = conv3d_para.shape
        stride = conv_3d.stride

        # reshape 3dconv
        app_para = conv3d_para[0:outplanes//2, :, :, :, :].view(outplanes//2, inplanes, 1, -1)
        motion_para = conv3d_para[outplanes//2:, :, :, :, :]
        padding_t = (app_para.size(-1) - 1)//2
        padding_s = (motion_para.size(-1) - 1)//2

        out_app = self.apperance_pretrain_temporal(x, app_para, padding_t, stride=stride)
        out_motion = F.conv3d(x, weight=motion_para, stride=stride, groups=groups, padding=(padding_t, padding_s, padding_s))

        out = torch.cat([out_app,  out_motion], dim=1)
        return out

    def apperance_pretrain_temporal(self, x, app_para, padding_t, groups=1, stride=(1, 1, 1)):

        b, c, t, h, w = x.size()
        app_kernel_chunks = torch.chunk(app_para, 2, dim=0)

        #reshape to column or row
        out_column = einops.rearrange(x, 'b  c  t  h  w  -> b c t  ( h w)', t=t, h=h, w=w)
        out_row = einops.rearrange(x, 'b  c  t  h  w  -> b  c t  ( w h)', t=t, h=h, w=w)

        # convolve
        out_column = F.conv2d(out_column, weight=app_kernel_chunks[0], stride=(stride[0], stride[1]), groups=groups, padding=(0, padding_t))
        out_row = F.conv2d(out_row, weight=app_kernel_chunks[1], stride=(stride[0], stride[1]), groups=groups, padding=(0, padding_t))

        # reshape back to original shape
        out_column = einops.rearrange(out_column, 'b  c  t ( h w)  -> b c t h w', t=t//stride[0], h=h//stride[1], w=w//stride[1])
        out_row = einops.rearrange(out_row, 'b  c  t ( w h)  -> b c t h w', t=t//stride[0], h=h//stride[1], w=w//stride[1])

        # concat along the channel dimension
        out_app = torch.cat([out_column, out_row], dim=1)
        return out_app



class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.

    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d_reshape' or self.conv_cfg['type'] == 'Conv2plus1d'

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x)

        return x