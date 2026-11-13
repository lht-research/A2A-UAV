import os as _os

_benchmark_root = _os.path.abspath(
    _os.environ.get('A2AUAV_BENCHMARK_ROOT', 'benchmark_inputs'))

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_scheme = 'single-class-uav'
custom_hooks = [
    dict(max_check_iters=500, strict=True, type='FiniteLossGuardHook'),
    dict(check_interval=50, max_seconds=28800, type='WallClockLimitHook'),
    dict(
        min_score_to_save=1e-12,
        monitor='monitor/bbox_mAP_50',
        out_dir=
        (_benchmark_root + '/panorama/runs/ddq'),
        type='ProtocolBestCheckpointHook'),
    dict(
        burn_in_iters=6000,
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
find_unused_parameters = True
gate1_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_gate1_ddq_virtual.json')
input_scale_policy = 'tile-native-1280'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
max_epochs = 12
metainfo = dict(classes=('uav', ))
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint=
            (_benchmark_root + '/panorama/pretrained/resnet50-11ad3fa6.pth'),
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='DDQDETRHead'),
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
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dense_topk_ratio=1.5,
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    dqs_cfg=dict(iou_threshold=0.8, type='nms'),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DDQDETR',
    with_box_refine=True)
monitor_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_val_tiles_monitor_virtual.json')
optim_wrapper = dict(
    accumulative_counts=2,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='ProtocolOptimWrapper')
overfit_ann_file = (_benchmark_root + '/panorama/data/coco_tiles/annotations/instances_train_tiles_overfit256_virtual.json')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=36000,
        gamma=0.1,
        milestones=[
            24000,
            32000,
        ],
        type='MultiStepLR'),
]
protocol = dict(
    batch_size_per_gpu=1,
    clip_grad_max_norm=0.1,
    effective_batch=8,
    eval_precision='bf16_or_fp32',
    family='ddq',
    grad_accum=2,
    lr=5e-05,
    train_precision='fp32',
    warmup_iters=500)
randomness = dict(deterministic=False, seed=20260610)
resume = False
run_name = 'ddq'
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
    (_benchmark_root + '/panorama/runs/ddq/eval/val_full'),
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
train_cfg = dict(max_iters=36000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=1,
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
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
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
work_dir = (_benchmark_root + '/panorama/runs/ddq')


env_cfg['dist_cfg']['timeout'] = 10800
test_evaluator['max_dets_per_image'] = 100
