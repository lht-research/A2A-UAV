import os as _os

_benchmark_root = _os.path.abspath(
    _os.environ.get('A2AUAV_BENCHMARK_ROOT', 'benchmark_inputs'))

dataset_type = 'CocoDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1706, 960), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1706, 960),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        (_benchmark_root + '/rgb/data/coco/annotations/instances_train_full.json'),
        img_prefix='',
        classes=('uav', ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1706, 960), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=
        (_benchmark_root + '/rgb/data/coco/annotations/instances_val_benchmark.json'),
        img_prefix='',
        classes=('uav', ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1706, 960),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True),
    test=dict(
        type='CocoDataset',
        ann_file=
        (_benchmark_root + '/rgb/data/coco/annotations/instances_test_rgb_single.json'),
        img_prefix='',
        classes=('uav', ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1706, 960),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.00125,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10000,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='RepairValidationGuard',
        monitor='bbox_mAP_50',
        patience=4,
        min_delta=0.005,
        max_empty_epochs=3,
        priority='LOWEST')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            (_benchmark_root + '/rgb/pretrained/resnet50-11ad3fa6.pth')
        )),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        refine=True,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSAlignHead',
        norm_cfg=None,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        fpn_layer='p2',
        fraction=0.5,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='AliTSM',
            ignore_iof_thr=-1,
            gpu_assign_thr=256,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='kl',
            assign_metric_m='iou',
            topk=[3, 1],
            ratio=0.9,
            alpha=1,
            beta=7,
            alpha_m=0.75),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
work_dir = (_benchmark_root + '/rgb/runs/fcos_kl')
gpu_ids = range(0, 2)
classes = ('uav', )
seed = 20260617
custom_imports = dict(imports=['repair_guard'], allow_failed_imports=False)
uav_uav_62_early_stop = dict(
    enabled=True,
    monitor='bbox_mAP_50',
    min_delta=0.005,
    patience=4,
    val_interval=1)
uav_uav_62_no_save_best_empty_eval_guard = dict(enabled=True)
uav_uav_62_oldstack = dict(
    run_name='exp_007_kldet',
    input_mode='whole-image',
    image_scale=(1706, 960),
    gpus='0,1',
    eff_batch=4,
    lr=0.00125)
