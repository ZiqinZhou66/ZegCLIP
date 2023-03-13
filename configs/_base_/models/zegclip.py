# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [5, 7, 11]

model = dict(
    type='ZegCLIP',
    pretrained='Path/to/pretrained/RN50.pt',
    context_length=5,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=21,
        style='pytorch'),
    decode_head=dict(
        type='ATMSingleHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        num_classes=150,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=in_channels // 2,
        loss_decode=dict(
            type='SegPlussLoss',  num_classes=150, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
    )
