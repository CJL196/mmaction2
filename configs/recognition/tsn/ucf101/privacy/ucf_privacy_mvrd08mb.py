_base_ = [
    '../../../../_base_/default_runtime.py'
]
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        # pretrained=None,
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='SigmoidHead',
        num_classes=5,
        in_channels=2048,
        multi_class=True,
        init_std=0.01,
        # loss_cls=dict(type='BCELossWithLogits'),
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        # mean=[33.92284211,33.92284211,33.92284211],
        # std=[46.85877167,46.85877167,46.85877167],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = f'/home/node1/Desktop/code/ai/data/ucf101/ucf_MvMbRd/ucf_a_0_8b'
data_root_val = data_root
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = 'data/ucf101/annotations/vp_ucf_train_split_01.txt'
ann_file_val = 'data/ucf101/annotations/vp_ucf_test_split_01.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=20,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:06}.png',
        pipeline=train_pipeline,
        multi_class=True, # 设置为True以支持多标签 
        num_classes=5        
        )
)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.png',
        pipeline=val_pipeline,
        test_mode=True,
        multi_class=True, # 设置为True以支持多标签 
        num_classes=5     
        )
)
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.png',
        pipeline=test_pipeline,
        test_mode=True,
        multi_class=True, # 设置为True以支持多标签 
        num_classes=5 
    )
)

# 评估指标使用MultiLabelMetric
val_evaluator = dict(type='MultiLabelMetric', metric_list=('cMAP', 'f1_score'))
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    checkpoint=dict(interval=10, max_keep_ckpts=1))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=12,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=100)
]

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=256)
log_config = dict(
    interval=1,  # 每个步骤记录一次
    hooks=[
        dict(type='TextLoggerHook'),  # 记录日志信息到文本
        dict(type='TensorboardLoggerHook'),  # 启用 TensorBoard
    ]
)
