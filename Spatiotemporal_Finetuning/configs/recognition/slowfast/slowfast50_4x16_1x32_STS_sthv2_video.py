_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        slow_pathway=dict(reshape_t=True, reshape_st=False),
        fast_pathway=dict(reshape_t=True, reshape_st=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=174,
        spatial_type='avg',
        dropout_ratio=0.5),
)
# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data1/data/20bn-something-something-v2'
data_root_val = '/data1/data/20bn-something-something-v2'
ann_file_train = '/data1/data/20bn-something-something-v2/train_video.txt'
ann_file_val = '/data1/data/20bn-something-something-v2/val_video.txt'
ann_file_test = '/data1/data/20bn-something-something-v2/val_video.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=32),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    #dict(type='Flip', flip_ratio=0),
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
        clip_len=1,
        frame_interval=1,
        num_clips=32,
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
        clip_len=1,
        frame_interval=1,
        num_clips=32,
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
    videos_per_gpu=8,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

total_epochs = 50
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 64 batchsize
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)

inflate = False

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
find_unused_parameters = False
work_dir = './work_dirs/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb'  # noqa: E501
#load_from =  '[your pre-trained weights path]'
checkpoint_config = dict(interval=5)
fp16 = dict(loss_scale=512.,)