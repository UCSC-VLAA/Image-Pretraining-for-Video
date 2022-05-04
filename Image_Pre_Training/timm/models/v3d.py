""" Vision Transformer (ViT) in PyTorch

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from timm.data import  IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .video_models import ResNet3dSlowFast, ResNet2Plus1d, X3D, ResNet3dPathway
from .video_models.resnet3d_slowonly import ResNet3dSlowOnly
from .registry import register_model
from .layers import  create_classifier
from .video_models.resnet_csn import ResNet3dCSN

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        # 'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
       'classifier': 'head',
        **kwargs
    }

default_urls = {
    'slowonly':{
        'k400': 'https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth',
        'k700': 'https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015-9250f662.pth',
        'img_k400': 'https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth',
},
    'ir_csn_50':{
        'k400': 'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb_20210618-4e29e2e8.pth',
        'ig65m': 'https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth'
},
    'slowfast':{
        'k400':'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth',
    },
    'r2plus1d34':{
        'k400': 'https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_kinetics400_rgb_20200618-63462eb3.pth'
    },
    'x3d':{
        'k400': 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
    }


}
default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'v3d_slowonly50_224': _cfg(
        url='https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth')
}

class V3D(nn.Module):

    def __init__(self, feature_extractor, num_classes=1000, pretrained=False, num_features=512, pretrain_dataset='k400', drop_rate=0., default_cfg=None,drop_path_rate=0.,
                 model_name=None, global_pool='avg'):
        super(V3D, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.default_cfg = default_cfg
        self.model_name = model_name
        if pretrained:
            self.checkpoint_or_url = default_urls[model_name][pretrain_dataset]
            self.pretrain_dataset = pretrain_dataset
        else:
            self.checkpoint_or_url = None

        self.get_feature = feature_extractor

        # Head + Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if self.model_name == 'x3d':
            _, self.fc1 = create_classifier(
                self.get_feature.feat_dim, num_features, pool_type=global_pool)
            self.relu = nn.ReLU(inplace=True)
            _, self.classifier = create_classifier(
                self.num_features, self.num_classes, pool_type=global_pool)

        else:
            _, self.classifier = create_classifier(
                self.num_features, self.num_classes, pool_type=global_pool)

        # if not pretrained:
        #     print('Using the one-init instead of zero-init')
        self.init_weights(pretrained)


    def init_weights(self, pretrained):
        if pretrained:
            self.get_feature._init_weights(self.get_feature, pretrained=self.checkpoint_or_url)
            if self.pretrain_dataset == 'imagenet':
                self.classifier.weight = load_state_dict_from_url(self.checkpoint_or_url)['fc.weight']
                self.classifier.bias =  load_state_dict_from_url(self.checkpoint_or_url)['fc.bias']
                _logger.info('loading the classifier successfully')

        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        _, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward(self, x):

        x = x.unsqueeze(2)
        x = self.get_feature(x)
        if isinstance(x, tuple):
           x =  torch.cat([self.global_pool(x[0]), self.global_pool(x[1])], dim=1)
           x = self.global_pool(x)
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
           return  x

        else:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if self.model_name == 'x3d':
            x = self.fc1(x)
            x = self.relu(x)
            x = self.classifier(x)
            return x
        else:
            return self.classifier(x)



def filter_pretrained_weight(pretrained_url, model):

    state_dict = load_state_dict_from_url(pretrained_url, progress=False, map_location='cpu')['state_dict']
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        k = k.replace('backbone.', '')
        if k in model_state_dict:
            model_state_dict[k] = v
        else:
            print('Not using the weight of: ', k)
    return model_state_dict



@register_model
def v3d_slowonly50_224(pretrained=False, **kwargs):
    """ v3d_slowonly50_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']
    pretrained2d = kwargs['pretrained2d']
    num_classes = kwargs['num_classes']
    drop_path_rate = kwargs['drop_path_rate']
    kwargs={}

    kwargs['pretrained2d'] = pretrained2d
    kwargs ['drop_path_rate'] = drop_path_rate
    slowonly50 = ResNet3dSlowOnly(depth=50, reshape_t=False, norm_eval=False, reshape_st=False, **kwargs)

    model = V3D(feature_extractor=slowonly50, num_features=2048, num_classes=num_classes, pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='slowonly',
                default_cfg=default_cfgs)
    return model



@register_model
def v3d_slowfast50_4x16_224(pretrained=False, **kwargs):
    """ v3d_slowfast50_4x16_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']

    slowfast50 = ResNet3dSlowFast(slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     reshape_t=False,
                     reshape_st=False,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     fusion_kernel=5,
                     inflate=(0, 0, 1, 1),
                     drop_path_rate=0.),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     reshape_t=False,
                     reshape_st=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                    drop_path_rate=0.))

    model = V3D(feature_extractor=slowfast50, num_features=2304, num_classes=kwargs['num_classes'], pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='slowfast',
                default_cfg=default_cfgs)
    return model

@register_model
def v3d_slowfast50_8x8_224(pretrained=False, **kwargs):
    """ v3d_slowfast50_8x8_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']

    slowfast50 = ResNet3dSlowFast(slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     reshape_t=False,
                     reshape_st=False,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     fusion_kernel=7,
                     inflate=(0, 0, 1, 1),
                     drop_path_rate=0.),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     reshape_t=False,
                     reshape_st=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                    drop_path_rate=0.))

    model = V3D(feature_extractor=slowfast50, num_features=2304, num_classes=kwargs['num_classes'], pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='slowfast',
                default_cfg=default_cfgs)
    return model

@register_model
def v3d_r2plus1d34_224(pretrained=False, **kwargs):
    """ v3d_r2plus1d34_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']

    args = dict(
        depth=34,
        pretrained=pretrained,
        pretrained2d=False,
        norm_eval=False,
        frozen_stages=0,
        conv_cfg=dict(type='Conv2plus1d'),
        norm_cfg=dict(type='BN3d', requires_grad=True, eps=1e-3),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 1, 1, 1),
        zero_init_residual=False,
        drop_path_rate = kwargs ['drop_path_rate'])

    resnet2plus1d = ResNet2Plus1d(**args)

    model = V3D(feature_extractor=resnet2plus1d, num_features=512, num_classes=kwargs['num_classes'], pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='r2plus1d34',
                default_cfg=default_cfgs)
    return model



@register_model
def v3d_csn50_ir_224(pretrained=False, **kwargs):
    """ v3d_csn50_ir_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']

    args = dict(
        pretrained=None,
        depth=50,
        norm_eval=False,
        bottleneck_mode='ir',
        with_pool2=False,
        reshape_t=False,
        reshape_st=False,
        zero_init_residual=False,
       # frozen_stages=4,
        drop_path_rate=kwargs['drop_path_rate'])
    csn50 = ResNet3dCSN(**args)


    model = V3D(feature_extractor=csn50, num_features=2048, num_classes=kwargs['num_classes'], pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='ir_csn_50',
                default_cfg=default_cfgs)
    return model




@register_model
def v3d_x3d_s_224(pretrained=False, **kwargs):
    """ v3d_x3d_s_224
    """
    pretrain_dataset = kwargs['pretrain_dataset']
    num_classes = kwargs['num_classes']

    args = dict(
        gamma_w=1, gamma_b=2.25, gamma_d=2.2)
    x3d = X3D(**args)

    model = V3D(feature_extractor=x3d, num_features=2048, num_classes=num_classes, pretrained=pretrained,
                pretrain_dataset=pretrain_dataset, model_name='x3d',
                default_cfg=default_cfgs)
    return model