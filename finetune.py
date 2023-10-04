import importlib
import random
import os
import datetime
import time
import pickle
import argparse

import pandas as pd
import numpy as np
import torch
import torchinfo
from torch import nn
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

import model_utils
import presets
from data import data_argument
from data import dataloader, dataloader2
from datamanagement import datasets
from models import nest, vit, resnet, dan

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', '-cp', type=str, required=True)
# Add an argument
parser.add_argument('--load_train_index', '-lti', type=str2bool, required=True)
parser.add_argument('--optimizer_type', '-ot', type=str, default='Adam', required=False)
parser.add_argument('--lr', '-lr',type=float, default=0.001, required=False)
parser.add_argument('--epoches', '-e', type=int, default=100, required=False)
parser.add_argument('--batch_size', '-bs', type=int, default=8, required=False)
parser.add_argument('--num_workers', '-nw', type=int, default=4, required=False)
parser.add_argument("--save_every_n", type=int, default=50, required=False)
parser.add_argument("--patience", type=int, default=20, required=False)
parser.add_argument("--train_transform", '-tt', nargs="+", default=['bias', 'noise', 'blur', 'flip', 'spatial', 'swap'], required=False)

parser.add_argument("--use_age", '-ua', type=int, default=0, required=False)
# Parse the argument
args = parser.parse_args()

# Readin original database and model setting
import pandas as pd
df2 = pd.read_csv(args.csv_path)

for k, row in df2.iterrows():
    result_dir = os.path.join('./result', row['result_dir'])
    if args.load_train_index:
        model_dir = os.path.join(result_dir, row['model_dir'])
    else:
        model_dir = result_dir

    print('------------------------------------')
    print(f'{k}/{len(df2)}: {result_dir}/{model_dir}')

    with open(os.path.join(result_dir,'db_setting.pkl'), 'rb') as f:
        db_setting = pickle.load(f)
    with open(os.path.join(result_dir,'model_setting.pkl'), 'rb') as f:
        model_setting = pickle.load(f)

    # Create new training setting
    train_setting = presets.get_train_setting(args.optimizer_type, args.lr,
                                                args.epoches, args.batch_size,
                                                args.num_workers, args.save_every_n, 
                                                args.patience, args.train_transform)

    train_transform = data_argument.build_transform(train_setting.train_transform, shape=model_setting.image_sizes)
    test_transform = data_argument.build_transform(train_setting.test_transform, shape=model_setting.image_sizes)

    # Load Subjects
    print('Loading Subjects')

    # for 10-fold, train test index is saved
    if args.load_train_index:
        with open(os.path.join(result_dir,'subjects.pkl'), 'rb') as f:
            subjects = pickle.load(f)
        with open(os.path.join(model_dir, f'train.pkl'), 'rb') as f:
            train_index = pickle.load(f)
        with open(os.path.join(model_dir, f'test.pkl'), 'rb') as f:
            test_index = pickle.load(f)

        total_train_subjects = np.array(subjects)[train_index].tolist()
        train_subjects, valid_subjects = train_test_split(total_train_subjects, test_size=0.05)
        test_subjects = np.array(subjects)[test_index].tolist()
    else:
        # for single dataset validation
        train_centers = []
        for train_path in db_setting.train_pathes:
            train_centers += datasets.load_dataset(train_path)
        test_centers = []
        for test_path in db_setting.test_pathes:
            test_centers += datasets.load_dataset(test_path)

        if not args.use_age:
            total_train_subjects = dataloader.load_subjects(train_centers, db_setting.img_path)
            train_subjects, valid_subjects = train_test_split(total_train_subjects, test_size=0.05)
            test_subjects = dataloader.load_subjects(test_centers, db_setting.img_path)
        else:
            total_train_subjects = dataloader2.load_subjects(train_centers, db_setting.img_path)
            train_subjects, valid_subjects = train_test_split(total_train_subjects, test_size=0.05)
            test_subjects = dataloader2.load_subjects(test_centers, db_setting.img_path)
        subjects = total_train_subjects + test_subjects

    for subject in tqdm(subjects):
        subject.load()

    if not args.use_age:
        train_dataset = dataloader.ArguDataset(train_subjects, train_transform)
        valid_dataset = dataloader.ArguDataset(valid_subjects, test_transform)
        test_dataset = dataloader.ArguDataset(test_subjects, test_transform)
    else:
        train_dataset = dataloader2.ArguDataset(train_subjects, train_transform)
        valid_dataset = dataloader2.ArguDataset(valid_subjects, test_transform)
        test_dataset = dataloader2.ArguDataset(test_subjects, test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_setting.batch_size,
                                            num_workers=train_setting.num_workers, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_setting.batch_size,
                                                num_workers=train_setting.num_workers, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_setting.batch_size,
                                                num_workers=train_setting.num_workers, shuffle=True)

    files = os.listdir(model_dir)

    # select best model
    best_acc = 0
    best_model_path = None
    for f in files:
        if 'Model' in f and '.pt' in f:
            acc_loc = f.find('acc')
            acc = float(f[acc_loc+3:acc_loc+7])
            print(acc)
            if acc > best_acc:
                best_model_path = f
                best_acc = acc

    # set output prefix
    i = 1
    largest_i = 0
    for f in files:
        if f'ft{i}' in f:
            i += 1
        if i > largest_i:
            largest_i = i
    prefix = f'ft{largest_i}'

    full_model_path = os.path.join(model_dir, best_model_path)
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
    # Set training helper
    if train_setting.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=train_setting.lr)

    # useless
    scheduler = MultiStepLR(optimizer, milestones=[300,400], gamma=0.1)

    criterion = nn.CrossEntropyLoss().cuda()

    # start training
    lowest_loss = 10000

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    print('Start training')
    with open(os.path.join(model_dir, f'{prefix}_log.txt'), 'w') as f:
        print(f'Model path:{full_model_path}', file=f)
        for epoch in range(train_setting.epoches):
            if not args.use_age:
                train_loss, train_acc, *_ = model_utils.train_model(model, train_loader, optimizer, scheduler, criterion, epoch, f=f)

                valid_loss, valid_acc, *_ = model_utils.test_model(model, valid_loader, criterion, phase='valid', f=f)
            else:
                train_loss, train_acc, *_ = model_utils.train_model2(model, train_loader, optimizer, scheduler, criterion, epoch, f=f)

                valid_loss, valid_acc, *_ = model_utils.test_model2(model, valid_loader, criterion, phase='valid', f=f)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_loss > lowest_loss:
                count += 1
            else:
                lowest_loss = valid_loss
                count = 0

            if count > train_setting.patience:
                print(f'Early Stopping in Epoch: {epoch}', file=f)
                break
            if epoch % train_setting.save_every_n == train_setting.save_every_n-1:
                if not args.use_age:
                    test, acc, *_ = model_utils.test_model(model, test_loader, criterion, phase='test', f=f)
                else:
                    test, acc, *_ = model_utils.test_model2(model, test_loader, criterion, phase='test', f=f)
                torch.save(model.state_dict(), os.path.join(model_dir, f'{prefix}_Model_acc{acc:.4f}_epoch{epoch}.pt'))

        df = pd.DataFrame(list(zip(train_losses, train_accs, valid_losses, valid_accs)),
                                columns =['Train_Loss', 'Train_Acc', 'Valid_Loss', 'Valid_Acc'])
        df.to_csv(os.path.join(model_dir, f'{prefix}_Train_Report.csv'))

        print(f'Timestamp: {time.time()}', file=f)
        if not args.use_age:
            test, acc, *_ = model_utils.test_model(model, test_loader, criterion, phase='test', f=f)
        else:
            test, acc, *_ = model_utils.test_model2(model, test_loader, criterion, phase='test', f=f)
        torch.save(model.state_dict(), os.path.join(model_dir, f'{prefix}_Model_acc{acc:.4f}.pt'))