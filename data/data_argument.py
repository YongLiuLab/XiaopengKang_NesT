import torchio as tio

rescale = tio.RescaleIntensity(out_min_max=(0, 1))
z_score = tio.ZNormalization(masking_method=lambda x: x > 0)

flip = tio.RandomFlip(axes=['LR'],
                      flip_probability=0.5)
elastic = tio.RandomElasticDeformation(p=0.5)

spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
    p = 0.75,
)

anisotropy = tio.RandomAnisotropy()

blur = tio.RandomBlur(p=0.2)

noise = tio.OneOf({
        tio.RandomNoise(mean=0, std=1): 0.5,
        tio.RandomNoise(mean=0, std=0.5): 0.5,
    },
    p = 1,
)
noise2 = tio.OneOf({
        tio.RandomNoise(mean=0, std=0.1): 0.5,
        tio.RandomNoise(mean=0, std=0.3): 0.5,
    },
    p = 1,
)

#noise = tio.RandomNoise(mean=0, std=1)

movement = tio.OneOf({
        tio.RandomGhosting(intensity=1): 0.5,
        tio.RandomMotion(num_transforms=6, image_interpolation='nearest'): 0.5,
    },
    p = 0.75,
)

bias_field = tio.RandomBiasField()
swap = tio.RandomSwap(patch_size=(6,8,6))

def build_transform(t_list, shape=(90, 112, 90)):
    trans_list = []
    for t in t_list:
        if t == 'rescale':
            trans_list.append(rescale)
        elif t == 'zscore':
            trans_list.append(z_score)
        elif t == 'cop':
            trans_list.append(tio.CropOrPad(shape))
        elif t == 'flip':
            trans_list.append(flip)
        elif t == 'elastic':
            trans_list.append(elastic)
        elif t == 'spatial':
            trans_list.append(spatial)
        elif t == 'anisotropy':
            trans_list.append(anisotropy)
        elif t == 'blur':
            trans_list.append(blur)
        elif t == 'noise':
            trans_list.append(noise)
        elif t == 'noise2':
            trans_list.append(noise2)
        elif t == 'movement':
            trans_list.append(movement)
        elif t == 'swap':
            trans_list.append(swap)
        elif t == 'bias':
            trans_list.append(bias_field)
    return tio.Compose(trans_list)