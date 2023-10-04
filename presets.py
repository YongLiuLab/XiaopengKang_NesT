import data.setting as ds
import models.setting as ms
import setting as ts

NEST2_XSMALL = ms.NestSetting(type='nest',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(12, 14, 12),
                            channels=1,
                            layer_heads=(8, 8, 8),
                            depthes=(6, 6, 8))

NEST2_TINY = ms.NestSetting(type='nest',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(6, 7, 6),
                            channels=1,
                            layer_heads=(8, 8),
                            depthes=(6, 6))

NEST2_SMALL = ms.NestSetting(type='nest',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(6, 7, 6),
                            channels=1,
                            layer_heads=(8, 8, 8),
                            depthes=(6, 6, 6))

NEST2_LARGE = ms.NestSetting(type='nest',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(6, 7, 6),
                            channels=1,
                            layer_heads=(8, 8, 8, 8),
                            depthes=(6, 6, 6, 6))

NEST4_TINY = ms.NestSetting(type='nest',
                            image_sizes=(48, 56, 48),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(6, 7, 6),
                            channels=1,
                            layer_heads=(8, 8),
                            depthes=(6, 6))

NEST4_SMALL = ms.NestSetting(type='nest',
                            image_sizes=(48, 56, 48),
                            patch_sizes=(2, 2, 2),
                            num_classes=2,
                            embed_dim=256,
                            mlp_mult=4,
                            dim_head=64,
                            dropout=0.5,
                            init_patch_embed_sizes=(6, 7, 6),
                            channels=1,
                            layer_heads=(8, 8, 8),
                            depthes=(6, 6, 6))

VIT4_TINY = ms.VitSetting(type='vit',
                            image_sizes=(48, 56, 48),
                            patch_sizes=(12, 14, 12),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=6,
                            depth=4,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT4_SMALL = ms.VitSetting(type='vit',
                            image_sizes=(48, 56, 48),
                            patch_sizes=(8, 8, 8),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=8,
                            depth=6,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT4_LARGE = ms.VitSetting(type='vit',
                            image_sizes=(48, 56, 48),
                            patch_sizes=(6, 7, 6),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=8,
                            depth=6,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT2_TINY = ms.VitSetting(type='vit',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(24, 28, 24),
                            num_classes=2,
                            embedding='linear',
                            embed_dim=256,
                            head=6,
                            depth=4,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT2_SMALL = ms.VitSetting(type='vit',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(16, 16, 16),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=8,
                            depth=6,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT2_LARGE = ms.VitSetting(type='vit',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(12, 14, 12),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=12,
                            depth=8,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT2_LARGE_NEW = ms.VitSetting(type='vit',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(12, 14, 12),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=256,
                            head=8,
                            depth=6,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

VIT2_XLARGE = ms.VitSetting(type='vit',
                            image_sizes=(96, 112, 96),
                            patch_sizes=(6, 7, 6),
                            num_classes=2,
                            embedding='conv',
                            embed_dim=128,
                            head=8,
                            depth=6,
                            mlp_mult=4,
                            dim_head=64,
                            channels=1,
                            dropout=0.5)

RESNET_34 = ms.ResNetSetting(type='resnet',
                            image_sizes=(96, 112, 96),
                            num_classes=2,
                            channels=1,
                            shortcut_type='A')

DAN_34 = ms.DanSetting(type='dan',
                        image_sizes=(96, 112, 96),
                        num_classes=2,
                        channels=1,
                        shortcut_type='A')

def get_db_setting(classes, dataset_path,
                   image_type, image_size, train_datasets, test_datasets):
    if image_type == 'G':
        img_path = [f'mri/mwp1t1_{image_size}mm.nii']
    elif image_type == 'W':
        img_path = [f'mri/mwp2t1_{image_size}mm.nii']
    elif image_type == 'GW':
        img_path = [f'mri/mwp1t1_{image_size}mm.nii',
                    f'mri/mwp2t1_{image_size}mm.nii']
    elif image_type == 'R':
        img_path = [f'rt1_{image_size}mm.nii']
    elif image_type == 'O':
        img_path = [f't1_{image_size}mm.nii']
    else:
        raise ValueError(f'image_type {image_type} is not support')

    return ds.DatabaseSetting(classes=classes,
                            dataset_path=dataset_path,
                            img_path=img_path,
                            train_datasets=train_datasets,
                            test_datasets=test_datasets)

def get_train_setting(optimizer_type,
                    lr,
                    epoches,
                    batch_size,
                    num_workers,
                    save_every_n,
                    patience,
                    train_transform):
    base_transform = ['cop', 'z_score']
    return ts.TrainSetting(optimizer_type=optimizer_type,
                            lr=lr,
                            epoches=epoches,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            save_every_n=save_every_n,
                            patience=patience,
                            train_transform=base_transform+train_transform,
                            test_transform=base_transform)

def get_model_setting(model):
    if model == 'NEST2_TINY':
        return NEST2_TINY
    elif model == 'NEST4_TINY':
        return NEST4_TINY
    elif model == 'NEST2_SMALL':
        return NEST2_SMALL
    elif model == 'NEST4_SMALL':
        return NEST4_SMALL
    elif model == 'NEST2_LARGE':
        return NEST2_LARGE
    elif model == 'VIT2_TINY':
        return VIT2_TINY
    elif model == 'VIT4_TINY':
        return VIT4_TINY
    elif model == 'VIT2_SMALL':
        return VIT2_SMALL
    elif model == 'VIT4_SMALL':
        return VIT4_SMALL
    elif model == 'VIT2_LARGE':
        return VIT2_LARGE
    elif model == 'VIT4_LARGE':
        return VIT4_LARGE
    elif model == 'RESNET_34':
        return RESNET_34
    elif model == 'DAN_34':
        return DAN_34
    elif model == 'VIT2_XLARGE':
        return VIT2_XLARGE
    elif model == 'VIT2_LARGE_NEW':
        return VIT2_LARGE_NEW
    elif model == 'NEST2_XSMALL':
        return NEST2_XSMALL