_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],
        mean = [11.56, 11.56, 11.56],
        #std=[58.395, 57.12, 57.375],
        std = [20.42, 20.42, 20.42],
        bgr_to_rgb=True,
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=18,
            # num_classes=3,
        ),
        mask_head=dict(
            num_classes=18,
            # num_classes=3,
        )),
    # model training and testing settings
)
dataset_type = 'str3dCocoDataset'
# data_root = 'data/scenecad/'
data_root = 'data/str3d/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Rotate',max_mag= 180.0, img_border_value=(0,0,0)),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # If you don't have a gt annotation, delete the python tools/train.py configs/balloon/mask-rcnn_r50-caffe_fpn_ms-poly-1x_balloon.py
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train1.json',
        # ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val1.json',
        # ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test1.json',
        # ann_file='annotations/val.json',
        data_prefix=dict(img='test/'),
        # data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val1.json',
    # ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
# test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000, val_interval=1)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test1.json',
    # ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(type='mmdet.DetLocalVisualizer',vis_backends=vis_backends,name='visualizer',save_dir='result')