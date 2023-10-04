import argparse
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch import nn
import pandas as pd

import model_utils
from data import data_argument
from data import dataloader, dataloader2
from datamanagement import datasets
from models import nest, vit, resnet, dan

model_dirs = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4',
              'fold5', 'fold6', 'fold7', 'fold8', 'fold9']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--in_path', '-ip', type=str, required=True)
    parser.add_argument('--out_path', '-op', type=str, required=True)
    parser.add_argument('--load_train_index', '-lti', type=str2bool, required=True)
    parser.add_argument("--use_age", '-ua', type=int, default=0, required=False)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    result_dict = {}
    for key, row in df.iterrows():
        result_dir = os.path.join('./result', row['result_dir'])

        # Readin original database and model setting
        with open(os.path.join(result_dir,'db_setting.pkl'), 'rb') as f:
            db_setting = pickle.load(f)
        with open(os.path.join(result_dir,'model_setting.pkl'), 'rb') as f:
            model_setting = pickle.load(f)
        with open(os.path.join(result_dir,'train_setting.pkl'), 'rb') as f:
            train_setting = pickle.load(f)
        
        if args.load_train_index:
            model_pathes = []
            for model_dir in model_dirs:
                model_path = os.path.join(result_dir, model_dir)
                if os.path.isdir(model_path):
                    model_pathes.append(model_path)
        else:
            model_pathes = [result_dir]

        accs = []
        for model_path in model_pathes:
            print(model_path)
            # for 10-fold, train test index is saved
            if args.load_train_index:
                with open(os.path.join(result_dir,'subjects.pkl'), 'rb') as f:
                    subjects = pickle.load(f)
                with open(os.path.join(model_path, f'test.pkl'), 'rb') as f:
                    test_index = pickle.load(f)
                test_subjects = np.array(subjects)[test_index].tolist()
            else:
                # single dataset validation
                test_centers = []
                for test_path in db_setting.test_pathes:
                    test_centers += datasets.load_dataset(test_path)

                if not args.use_age:
                    test_subjects = dataloader.load_subjects(test_centers, db_setting.img_path)
                else:
                    test_subjects = dataloader2.load_subjects(test_centers, db_setting.img_path)

            for subject in test_subjects:
                subject.load()
            test_transform = data_argument.build_transform(train_setting.test_transform, shape=model_setting.image_sizes)

            if not args.use_age:
                test_dataset = dataloader.ArguDataset(test_subjects, test_transform)
            else:
                test_dataset = dataloader2.ArguDataset(test_subjects, test_transform)

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
            if model_setting.type == 'nest':
                model = nest.NesT(model_setting).cuda()
                if args.use_age:
                    model = nest.NesT2(model_setting).cuda()
            elif model_setting.type == 'vit':
                model = vit.Vit(model_setting).cuda()
            elif model_setting.type == 'resnet':
                model = resnet.resnet34(num_classes=model_setting.num_classes, 
                                        shortcut_type=model_setting.shortcut_type).cuda()
            elif model_setting.type == 'dan':
                model = dan.dan34(num_classes=model_setting.num_classes, 
                                    shortcut_type=model_setting.shortcut_type).cuda()

            device = torch.device('cuda:0')

            model.load_state_dict(torch.load(full_model_path, map_location=device))

            criterion = nn.CrossEntropyLoss().cuda()

            if not args.use_age:
                _, acc, *_ = model_utils.test_model(model, test_loader, criterion, phase='test')
            else:
                _, acc, *_ = model_utils.test_model2(model, test_loader, criterion, phase='test')
            
            accs.append(acc)

        mean = np.mean(accs)
        std = np.std(accs)
        accs += [mean, std]

        result_dict[row['result_dir']] = accs
    save_df = pd.DataFrame.from_dict(result_dict, orient='index',
                                     columns=model_pathes+['Mean', 'Std'])
    save_df.to_csv(args.out_path)