_base_ = [
    '../../_base_/models/mvit_small.py', '../../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        # type='adapter_MViT',
        # type='MViT',

        # convmvit=True,
        # convdep=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth',  # noqa: E501
            prefix='backbone.')
            ),
    cls_head=dict(num_classes=101))


# dataset settings
dataset_type = 'RawframeDataset'
MyPrefix=''
# data_root = f'{MyPrefix}data/ucf101/videos'
data_root = '/home/node1/Desktop/code/ai/data/ucf101/ucf_MvMbRd/ucf_a_0_8b'
data_root_val = '/home/node1/Desktop/code/ai/data/ucf101/ucf_MvMbRd/ucf_a_0_8b'
# split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/home/node1/Desktop/code/ai/data/ucf101/ucf_MvMbRd/ann/train_ann.txt'
ann_file_val = f'/home/node1/Desktop/code/ai/data/ucf101/ucf_MvMbRd/ann/test_ann.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16, test_mode=True),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=1,test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(
    #     type='SampleFrames',
    #     clip_len=16,
    #     frame_interval=3,
    #     num_clips=5,
    #     test_mode=True),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=3,test_mode=True),
    # dict(type='UniformSample', clip_len=16,test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:06}.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.png',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.png',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1.6e-3
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=12,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=12,
        end=100,
        convert_to_iter_based=True)
]

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
#     # paramwise_cfg=dict( 
#     #         custom_keys={'backbone': dict(lr_mult=0, decay_mult=1.0),
#     #                      'localconv': dict(lr_mult=1, decay_mult=1.0),
#     #                     'myfuse': dict(lr_mult=1, decay_mult=1.0),
#     #                     'S_Adapter': dict(lr_mult=1, decay_mult=1.0),
#     #                     'T_Adapter': dict(lr_mult=1, decay_mult=1.0),
#     #                     'cls_token': dict(lr_mult=1, decay_mult=1.0),
#     #                      }),
#     clip_grad=dict(max_norm=40, norm_type=2))

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         by_epoch=True,
#         begin=0,
#         end=12,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=100,
#         eta_min=0,
#         by_epoch=True,
#         begin=0,
#         end=100)
# ]

# base_lr = 1.6e-2
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
#     paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         by_epoch=True,
#         begin=0,
#         end=30,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=100,
#         eta_min=base_lr / 100,
#         by_epoch=True,
#         begin=30,
#         end=100,
#         convert_to_iter_based=True)
# ]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=1), logger=dict(interval=200))

randomness=dict(seed=3407)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
