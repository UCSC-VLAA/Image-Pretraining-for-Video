_base_ = [
    '../../_base_/models/slowonly_r50.py',
    '../../_base_/schedules/sgd_150e_warmup.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(with_pool1=True, reshape_t=True, pretrained=None, reshape_st=False), cls_head=dict(num_classes=174))

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
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
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
        num_clips=8,
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
        num_clips=8,
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
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5)
total_epochs = 50


# runtime settings
work_dir = './work_dirs/slowonly_r50_8x4x1_64e_sthv2_rgb'
checkpoint_config = dict(interval=5)
fp16 = dict(loss_scale=512.,)
inflate = True
#load_from = '[your pre-trained weight path]'
