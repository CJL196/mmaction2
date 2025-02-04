default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=3, save_best='auto', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='SoftmaxHead',
        num_classes=13,
        in_channels=2048,
        init_std=0.01,
        average_clips='score'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'RawframeDataset'
data_root = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
data_root_val = data_root
ann_file_train = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/trainPrivacyRaw.txt'
ann_file_val = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/valPrivacyRaw.txt'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=6),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.5),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=6,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
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
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
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
        type='RawframeDataset',
        ann_file=ann_file_train,
        data_prefix=dict(
            img=
            data_root
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=6),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.5),
                random_crop=False,
                max_wh_scale_gap=1),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='PackActionInputs')
        ]))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RawframeDataset',
        ann_file=ann_file_val,
        data_prefix=dict(
            img=
            data_root_val
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=6,
                test_mode=True),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RawframeDataset',
        ann_file=ann_file_val,
        data_prefix=dict(
            img=
            data_root_val
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=25,
                test_mode=True),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='TenCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2),
    type='AmpOptimWrapper',
    loss_scale='dynamic')
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=50)
]
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
randomness = dict(seed=3407)
auto_scale_lr = dict(enable=False, base_batch_size=256)
launcher = 'pytorch'
work_dir = '/home/node1/Desktop/code/ai/open-mmlab/mmaction2/work_dirs/resnet_cls_softmax_privacy_08-cjl2-rd08mb'
log_processor = dict(
    custom_cfg=[dict(data_src='top1_acc', 
                     log_name='top1_acc_avg', 
                     method_name='mean', 
                     window_size='global')]
)