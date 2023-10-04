import argparse
import os
import pickle
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch import nn

import interpret_utils
import model_utils
from data import data_argument
from data import dataloader
from datamanagement import datasets, mask
from models import nest
from models import vit

if __name__ == '__main__':
    m = mask.NiiMask('./assets/atlas/BN_Atlas_274_combined_2mm.nii')

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--result_dir', '-rd', type=str, required=True)
    parser.add_argument('--load_train_index', '-lti', type=bool, required=True)
    parser.add_argument('--out_dir', '-od', type=str, required=True)
    args = parser.parse_args()

    # Readin original database and model setting
    result_dir = args.result_dir
    with open(os.path.join(result_dir,'db_setting.pkl'), 'rb') as f:
        db_setting = pickle.load(f)
    with open(os.path.join(result_dir,'model_setting.pkl'), 'rb') as f:
        model_setting = pickle.load(f)
    with open(os.path.join(result_dir,'train_setting.pkl'), 'rb') as f:
        train_setting = pickle.load(f)
    with open(os.path.join(result_dir,'subjects.pkl'), 'rb') as f:
        subjects = pickle.load(f)
    
    if args.load_train_index:
        model_dirs = os.listdir(args.result_dir)
        model_pathes = []
        for model_dir in model_dirs:
            model_path = os.path.join(args.result_dir, model_dir)
            if os.path.isdir(model_path):
                model_pathes.append(model_path)
    else:
        model_pathes = [args.result_dir]

    for model_path in model_pathes:
        print(model_path)
        # for 10-fold, train test index is saved
        if args.load_train_index:
            with open(os.path.join(model_path, f'test.pkl'), 'rb') as f:
                test_index = pickle.load(f)
            test_subjects = np.array(subjects)[test_index].tolist()
        else:
            # single dataset validation
            test_centers = []
            for test_path in db_setting.test_pathes:
                test_centers += datasets.load_dataset(test_path)

            test_subjects = dataloader.load_subjects(test_centers, db_setting.img_path)

        for subject in test_subjects:
            subject.load()
        test_transform = data_argument.build_transform(train_setting.test_transform, shape=model_setting.image_sizes)
        test_dataset = dataloader.ArguDataset(test_subjects, test_transform)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_setting.batch_size,
                                                    num_workers=train_setting.num_workers, shuffle=True)

        files = os.listdir(model_path)

        # select best model
        best_acc = 0
        best_model_path = None
        largest_i = 0
        for f in files:
            if 'Model' in f and '.pt' in f:
                acc_loc = f.find('acc')
                acc = float(f[acc_loc+3:acc_loc+7])
                print(acc)
                if acc > best_acc:
                    best_model_path = f
                    best_acc = acc

        full_model_path = os.path.join(model_path, best_model_path)
        print(f'Model path:{full_model_path}')

        # Load best Model
        print('Loading Model')
        target_layers = []
        if model_setting.type == 'nest':
            model_setting.dropout = 0
            model = nest.NesT(model_setting).cuda()
            for i in range(len(model.layers)-1):
                # last layer conv_norm_max would not run
                target_layers.append(model.layers[i][1])
            #target_layers += [model.mlp_head[0]]
            reshape = None
        elif model_setting.type == 'vit':
            model_setting.dropout = 0
            model = vit.Vit(model_setting).cuda()
            #for layer in model.transformer.layers:
            target_layers.append(model.transformer.layers[-1][1].norm)
            p1, p2, p3 = model.patch_sizes
            a = int(model.image_sizes[0] / p1)
            b = int(model.image_sizes[1] / p2)
            c = int(model.image_sizes[2] / p3)
            reshape = partial(interpret_utils.vit_reshape, height=a, width=b, depth=c)

        device = torch.device('cuda:0')
        model.load_state_dict(torch.load(full_model_path, map_location=device))

        interpret_utils.generate_cam_map(test_loader, model, target_layers, m, args.out_dir, reshape)

