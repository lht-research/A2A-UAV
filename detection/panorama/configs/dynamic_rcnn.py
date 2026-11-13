import os as _os

_benchmark_root = _os.path.abspath(
    _os.environ.get('A2AUAV_BENCHMARK_ROOT', 'benchmark_inputs'))

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_scheme = 'single-class-uav'
custom_hooks = [
    dict(max_check_iters=500, strict=True, type='FiniteLossGuardHook'),
    dict(check_interval=50, max_seconds=21600, type='WallClockLimitHook'),
    dict(
        min_score_to_save=1e-12,
        monitor='monitor/bbox_mAP_50',
        out_dir=
        (_benchmark_root + '/panorama/runs/dynamic_rcnn'),
        type='ProtocolBestCheckpointHook'),
    dict(
        burn_in_iters=4000,
        collapse_after_iter=4000,
        collapse_monitor='monitor/probe_max_score',
        collapse_threshold=0.001,
        hard_fail_iter=12000,
        min_delta=0.0,
        monitor='monitor/bbox_mAP_50',
        patience=3,
        start_threshold=0.02,
        type='ProtocolEarlyStopHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'lib.uav_coco_tile_dataset',
        'lib.tile_loading',
        'lib.wall_clock_hook',
        'lib.panorama_merge_metric',
        'lib.protocol_runtime',
    ])
data_root = 'data/coco/'
dataset_type = 'UavCocoTileDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=2000, max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
experiment_root = (_benchmark_root + '/panorama')
gate1_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_gate1_dynamic_rcnn_virtual.json')
input_scale_policy = 'tile-native-1280'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
metainfo = dict(classes=('uav', ))
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint=
            (_benchmark_root + '/panorama/pretrained/resnet50-11ad3fa6.pth'),
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='DynamicRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.33,
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                4,
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=300,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.85, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            dynamic_rcnn=dict(
                beta_topk=10,
                initial_beta=1.0,
                initial_iou=0.4,
                iou_topk=75,
                update_iter_interval=100),
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=300,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.85, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
monitor_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_monitor_virtual.json')
optim_wrapper = dict(
    accumulative_counts=2,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='ProtocolOptimWrapper')
overfit_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_overfit256_virtual.json')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=24000,
        gamma=0.1,
        milestones=[
            16000,
            22000,
        ],
        type='MultiStepLR'),
]
protocol = dict(
    batch_size_per_gpu=2,
    clip_grad_max_norm=1.0,
    effective_batch=8,
    eval_precision='bf16_or_fp32',
    family='dynamic_rcnn',
    grad_accum=2,
    lr=0.01,
    train_precision='fp32',
    warmup_iters=500)
randomness = dict(deterministic=False, seed=20260610)
resume = False
run_name = 'dynamic_rcnn'
seed = 20260610
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_full_virtual.json'),
        data_prefix=dict(img=''),
        metainfo=dict(classes=('uav', )),
        pipeline=[
            dict(to_float32=True, type='LoadPanoramaTileFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'sample_id',
                    'source_image_id',
                    'source_file_name',
                    'tile_x',
                    'tile_y',
                    'tile_size',
                    'source_width',
                    'source_height',
                    'virtual_tile',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='UavCocoTileDataset'),
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    (_benchmark_root + '/panorama/data/panorama_coco/annotations/instances_val_panorama_single.json'),
    iou_thr=0.55,
    outfile_prefix=
    (_benchmark_root + '/panorama/runs/dynamic_rcnn/eval/val_full'),
    prefix='panorama',
    type='PanoramaMergeCocoMetric')
test_full_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_test_tiles_full_virtual.json')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
tile_size = 1280
train_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_neg20_virtual.json')
train_cfg = dict(max_iters=24000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file=
        (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_neg20_virtual.json'),
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=2),
        metainfo=dict(classes=('uav', )),
        pipeline=[
            dict(to_float32=True, type='LoadPanoramaTileFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'sample_id',
                    'source_image_id',
                    'source_file_name',
                    'tile_x',
                    'tile_y',
                    'tile_size',
                    'source_width',
                    'source_height',
                    'virtual_tile',
                ),
                type='PackDetInputs'),
        ],
        type='UavCocoTileDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
train_precision = 'fp32'
val_benchmark_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_benchmark.json')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_monitor_virtual.json'),
        data_prefix=dict(img=''),
        metainfo=dict(classes=('uav', )),
        pipeline=[
            dict(to_float32=True, type='LoadPanoramaTileFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'sample_id',
                    'source_image_id',
                    'source_file_name',
                    'tile_x',
                    'tile_y',
                    'tile_size',
                    'source_width',
                    'source_height',
                    'virtual_tile',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='UavCocoTileDataset'),
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_monitor_virtual.json'),
    backend_args=None,
    metric='bbox',
    prefix='monitor',
    probe_image_ids=[
        387,
        567,
        783,
        1881,
        1935,
        2313,
        2547,
        2601,
    ],
    type='MonitorCocoMetric')
val_full_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_full_virtual.json')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = (_benchmark_root + '/panorama/runs/dynamic_rcnn')
