import os

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from grad_cam import grad_cam
from datamanagement import utils

def generate_cam_map(data_loader, model, target_layers, mask, out_path,
                    reshape_transform=None):
    cam = grad_cam.GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
    for inputs, _labels, names in data_loader:
        inputs = inputs.cuda()
        _labels = _labels.cuda()

        batch_cam_map = cam(input_tensor=inputs)
        for cam_map, name in zip(batch_cam_map, names):
            new_nii = utils.gen_nii(cam_map, mask.nii)
            new_nii = resample_to_img(new_nii, mask.nii, interpolation='nearest')
            # Apply brain mask to keep only brain area
            new_array = np.multiply(np.array(new_nii.dataobj), mask.data>0)
            utils.gen_nii(new_array, mask.nii, os.path.join(out_path, f'{name}.nii.gz'))

def vit_reshape(tensor, height=8, width=8, depth=8):
    # starting from 1 cause 0 are class token
    # height*width*depth = seq_len
    result = tensor[:, 1: , :].reshape(tensor.size(0),
        height, width, depth, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(3, 4).transpose(2, 3).transpose(1, 2)
    return result