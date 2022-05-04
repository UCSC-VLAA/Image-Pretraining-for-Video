_base_ = [
    '../../_base_/models/r2plus1d_r34.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(act_cfg=dict(type='ReLU'),  conv_cfg=dict(type='Conv2plus1d_reshape'),
                           temporal_strides=(1, 1, 1, 1),
                           norm_cfg=dict(type='BN3d', requires_grad=True, eps=1e-3)))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data1/data/kinetics_400/videos_320'
data_root_val = '/data1/data/kinetics_400/videos_320'
ann_file_train = '/data1/data/kinetics_400/train_new.txt'
ann_file_val = '/data1/data/kinetics_400/val_new.txt'
ann_file_test = '/data1/data/kinetics_400/val_new.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))
# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for batchsize 64
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# runtime settings
work_dir = './work_dirs/r2plus1d_r34_video_3d_8x8x1_180e_kinetics400_rgb/'


load_from = '/data1/lxh/save/output/r2+1d_34_baseline_a2_v3d_300/model_best.pth.tar' # change the path to your pre-trained weight
total_epochs = 50
checkpoint_config = dict(interval=5)
inflate = False
precise_bn = dict(num_iters=200) # pre_bn may slow down your training
find_unused_parameters = False
#fp16
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2),
                        )
fp16 = dict(loss_scale=512.,)
