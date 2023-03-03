# 模型的配置文件

model_cfg = dict(
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=311,  # 数据集类别大小
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))

# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='const',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size=128,  # 每个batch所含的训练样本数目
    num_workers=10,  # 使用多少个子进程加载数据
    train = dict(
        pretrained_flag = True,  # 是否使用预训练权重
        pretrained_weights = 'datas/mobilenet_v3_small-8427ecf0.pth',
        freeze_flag = False,  # 是否选择冻结训练
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = 'logs/MobileNetV3/2023-03-02-13-52-41/Last_Epoch100.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# optimizer
optimizer_cfg = dict(
    type='RMSprop',
    lr=0.001,  # 初始学习率，根据自己的batch size调试
    alpha=0.9,
    momentum=0.9,
    eps=0.0316,
    weight_decay=1e-5)

# learning 
lr_config = dict(type='StepLrUpdater', step=2, gamma=0.973, by_epoch=True)
#lr_config = dict(type='StepLrUpdater', warmup='linear', warmup_iters=500, warmup_ratio=0.25,step=[30,60,90])

