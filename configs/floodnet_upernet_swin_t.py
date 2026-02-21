custom_imports = dict(imports=['mmseg.datasets.floodnet'], allow_failed_imports=False)

dataset_type = 'FloodNetDataset'
data_root = '/data/FloodNet-Supervised_v1.0/'
num_classes = 10
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,  # try 8–16 per GPU on A100 40GB; adjust by your VRAM
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='train/train-org-img', seg_map_path='train/train-label-img'),
        img_suffix='.jpg', seg_map_suffix='_lab.png',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=False,
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='val/val-org-img', seg_map_path='val/val-label-img'),
        img_suffix='.jpg', seg_map_suffix='_lab.png',
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ----- Swin + UPerNet -----
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='SegDataPreProcessor',
                           mean=[123.675,116.28,103.53],
                           std=[58.395,57.12,57.375],
                           bgr_to_rgb=True),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
        window_size=7, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3,
        patch_norm=True, out_indices=(0,1,2,3),
        init_cfg=dict(type='Pretrained',
                      checkpoint='pretrained/swin_tiny_patch4_window7_224.pth'),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96,192,384,768], in_index=[0,1,2,3],
        pool_scales=(1,2,3,6), channels=256, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384, in_index=2, channels=128, num_convs=1,
        concat_input=False, dropout_ratio=0.1, num_classes=num_classes,
        norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4),
    ),
    train_cfg=dict(), test_cfg=dict(mode='whole'),
)

# ----- Optim, schedule, AMP(bf16) -----
optim_wrapper = dict(
    type='AmpOptimWrapper',  # enables mixed precision
    dtype='bfloat16',        # A100: prefer bf16
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9,0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }),
)

param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=1.0, begin=0, end=160000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=3),
)

# optional compile for PyTorch 2.x speed
env_cfg = dict(cudnn_benchmark=True)
default_scope = 'mmseg'
