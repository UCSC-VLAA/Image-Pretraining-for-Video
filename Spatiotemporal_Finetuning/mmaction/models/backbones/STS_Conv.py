import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

try:
    import einops

    has_einops = True
except ImportError:
    has_einops = False




def STS_3dconv(conv3d, x):
    """.

    Args:
        conv3d (ConvModule): 3D conv used for separating.
        x (tensor): input feature map.
    """
    if isinstance(conv3d, nn.Sequential):
        conv3d = conv3d[-1]   # for csn

    assert isinstance(conv3d, ConvModule), 'conv3d should be a ConvModule'

    # extract the parameters of conv3d
    conv3d_para = conv3d.conv.weight
    outplanes, inplanes, k_t, k_s, _ = conv3d_para.shape
    stride_t, stride_s, _ = conv3d.stride
    assert stride_t == 1, 'We dont support temporal downsampling currently'
    padding_t, padding_s = (k_t - 1)//2, (k_s - 1)//2
    groups = conv3d.groups

    # separate 3d conv into dynamic and static channels
    app_para, motion_para = torch.chunk(conv3d_para, 2, dim=0)


    if k_s > 1: # this is not a temporal 3d conv
        assert groups > 1, 'We only support depthwise 3D conv currently (3x3x3)'
        x_app, x_motion = torch.chunk(x, 2, dim=1)

        # static appearance modeling by splitting 3D kernel
        out_app = apperance_st(x_app, app_para, stride=(stride_t, stride_s), groups=groups // 2)

        # dynamic motion modeling by normal 3D conv
        out_motion = F.conv3d(x_motion, weight=motion_para, stride=(stride_t, stride_s, stride_s), groups=groups // 2, padding=(padding_t, padding_s, padding_s))

    else:
        assert groups == 1, 'We only support fully-connected temporal convolution (3x1x1)'

        # static appearance modeling by splitting 3D kernel
        out_app = apperance_t(x, app_para, padding_t=padding_t, groups=groups)

        # dynamic motion modeling by normal 3D conv
        out_motion = F.conv3d(x, weight=motion_para, stride=1, groups=groups, padding=(padding_t, padding_s, padding_s))

    # concat the feature map
    out = torch.cat([out_app, out_motion], dim=1)
    if conv3d.norm_cfg is not None:
        out = conv3d.norm(out)
    if conv3d.act_cfg is not None:
        out = conv3d.activate(out)
    return out



def apperance_st(x, app_para, stride=(1, 1), groups=1):
    """.

        Args:
            x (tensor): input feature map with 1/2 channels.
            app_para (weight): weights used for appearance modeling.
        """

    # prepare parameters
    b, c, t, h, w = x.size()
    outplanes, inplanes, _, _, _ = app_para.shape

    # middle 2D kernel 1x3x3
    spatial_para = app_para[:, :, 1:2, :, :]
    out_s = F.conv3d(x, weight=spatial_para, stride=(stride[0], stride[1], stride[1]), groups=groups, padding=(0, (spatial_para.size(-1) - 1) // 2, (spatial_para.size(-1) - 1) // 2))

    # t1 kernel reshapes into 1D (1x9), then convolve along the row direction
    # note that some backbones use temporal dowsampling, we don't support temporal downsampling by using 1D conv
    row_para = app_para[:, :, 0, :, :].view(outplanes, inplanes,  -1)
    out_row = einops.rearrange(x, 'b c t h w  -> (b t) c (h w)', t=t, h=h, w=w)
    out_row = F.conv1d(out_row, weight=row_para, stride=(stride[1]*stride[1]), groups=groups, padding=((row_para.size(-1) - 1) // 2))
    out_row = einops.rearrange(out_row, '(b t) c (h w)  -> b c t h w', t=t, h=h // stride[1], w=w // stride[1])

    # t2 kernel reshapes into 1D (9x1), then convolve along the column direction
    column_para = app_para[:, :, 2, :, :].view(outplanes, inplanes,  -1)
    out_column = einops.rearrange(x, 'b c t h w  -> (b t) c (w h)', t=t, h=h, w=w)
    out_column = F.conv1d(out_column, weight=column_para, stride=(stride[1]*stride[1]), groups=groups, padding=((column_para.size(-1) - 1) // 2))
    out_column = einops.rearrange(out_column, '(b t) c (w h)  -> b c t h w', t=t, h=h // stride[1], w=w // stride[1])

    out = out_s + out_row + out_column
    return out





def apperance_t( x, app_para, padding_t, groups=1):
    """.

        Args:
            x (tensor): input feature map with 1/2 channels.
            app_para (weight): weights used for appearance modeling.
            padding_t (int): temporal padding for maintaining the same size of feature map
        """

    # prepare parameters
    b, c, t, h, w = x.size()
    app_para = app_para.view(app_para.size(0), app_para.size(1), -1)
    app_kernel_chunks = torch.chunk(app_para, 2, dim=0)


    # reshape to column or row
    out_column = einops.rearrange(x, 'b  c  t  h  w  -> (b t) c ( h w)', t=t, h=h, w=w)
    out_row = einops.rearrange(x, 'b  c  t  h  w  -> (b t) c ( w h)', t=t, h=h, w=w)

    # convolve
    out_column = F.conv1d(out_column, weight=app_kernel_chunks[0], stride=1, groups=groups, padding=padding_t)
    out_row = F.conv1d(out_row, weight=app_kernel_chunks[1], stride=1, groups=groups, padding=padding_t)

    # reshape back to original shape
    out_column = einops.rearrange(out_column, '(b t) c ( h w)  -> b c t h w', t=t, h=h, w=w)
    out_row = einops.rearrange(out_row, '(b t) c ( w h)  -> b c t h w', t=t, h=h, w=w)

    # concat along the channel dimension
    out_app = torch.cat([out_column, out_row], dim=1)
    return out_app


