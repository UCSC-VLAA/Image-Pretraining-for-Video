_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']



model = dict( backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2, reshape_st=True),)
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
    dict(type='SampleFrames', clip_len=13, frame_interval=6, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 192), lazy=True),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(160, 160), keep_ratio=False, lazy=True),
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
        clip_len=13,
        frame_interval=6,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 192), lazy=True),
    dict(type='CenterCrop', crop_size=160, lazy=True),
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
        clip_len=13,
        frame_interval=6,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 192)),
    dict(type='ThreeCrop', crop_size=192),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=1,
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
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# total_epochs = 50
optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9,
    weight_decay=1e-4,
    paramwise_cfg = dict(norm_decay_mult=0.,
                         bias_decay_mult=0.,
                         ))  # this lr is used for 8 gpus

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
     warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)
# lr_config = dict(policy='step', step=[5, 10])
total_epochs = 100
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
checkpoint_config = dict(interval=5)
find_unused_parameters = False
work_dir = './work_dirs/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb'
#load_from = '[your pre-trained weight path]'
inflate = False
precise_bn = dict(num_iters=200)

#fp16
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2),
                        )
fp16 = dict(loss_scale=512.,)