_base_ = [
    '../../_base_/models/c3d_sports1m_pretrained.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'RawframeDataset'
MyPrefix=''
# data_root = f'{MyPrefix}data/ucf101/videos'
data_root = '/data/zhengqingfeng/ucf101_/feature_map_dir/mvd_filter'
data_root_val = '/data/zhengqingfeng/ucf101_/feature_map_dir/mvd_filter'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'{MyPrefix}data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'{MyPrefix}data/ucf101/ucf101_val_split_{split}_rawframes.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=16, test_mode=True),
    dict(type='SampleFrames', clip_len=16, frame_interval=3, num_clips=1,test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
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
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
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
        pipeline=train_pipeline))
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
    type='EpochBasedTrainLoop', max_epochs=45, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    # paramwise_cfg=dict( 
    #         custom_keys={'backbone': dict(lr_mult=0, decay_mult=1.0),
    #                      'localconv': dict(lr_mult=1, decay_mult=1.0),
    #                     'myfuse': dict(lr_mult=1, decay_mult=1.0),
    #                     'S_Adapter': dict(lr_mult=1, decay_mult=1.0),
    #                     'T_Adapter': dict(lr_mult=1, decay_mult=1.0),
    #                     'cls_token': dict(lr_mult=1, decay_mult=1.0),
    #                      }),
    clip_grad=dict(max_norm=40, norm_type=2))

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
        begin=12,
        end=100)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=1), logger=dict(interval=200))

randomness=dict(seed=3407)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)

