dataset_type = 'str3dCocoDataset'
data_root = 'data/scenecad/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=(256, 256),
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(256, 256),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='FilterAnnotations', min_gt_bbox_wh=(1e-05, 1e-05), by_mask=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='str3dCocoDataset',
        data_root='data/scenecad/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/', seg='annotations/panoptic_train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomResize',
                scale=(256, 256),
                ratio_range=(0.1, 2.0),
                resize_type='Resize',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(256, 256),
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1e-05, 1e-05),
                by_mask=True),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='str3dCocoDataset',
        data_root='data/scenecad/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
            dict(type='Resize', scale=(256, 256), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='str3dCocoDataset',
        data_root='data/scenecad/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
            dict(type='Resize', scale=(256, 256), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/scenecad/annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/scenecad/annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=200),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='WandbVisBackend')],
    name='visualizer',
    save_dir='result')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
image_size = (256, 256)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=(256, 256),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[11.56, 11.56, 11.56],
    std=[20.42, 20.42, 20.42],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=255,
    batch_augments=[
        dict(
            type='BatchFixedSizePad',
            size=(256, 256),
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=False)
    ])
num_things_classes = 1
num_stuff_classes = 0
num_classes = 1
model = dict(
    type='Mask2Former',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[11.56, 11.56, 11.56],
        std=[20.42, 20.42, 20.42],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
        seg_pad_value=255,
        batch_augments=[
            dict(
                type='BatchFixedSizePad',
                size=(256, 256),
                img_pad_value=0,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=False)
        ]),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=1,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8, dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=1,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True),
    init_cfg=None)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0)),
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))
max_iters = 368750
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=368750,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)
interval = 5000
dynamic_intervals = [(365001, 368750)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=16)
launcher = 'none'
work_dir = ''
