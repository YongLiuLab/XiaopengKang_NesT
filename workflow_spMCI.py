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
import torchio as tio

import model_utils
import presets
from data import data_argument
from datamanagement import datasets
from data import dataloader, dataloader2
from models import nest, vit, resnet, dan

# load sMCI and pMCI
def load_subjects(centers, label_csv_path, img_pathes=['mri/mwp1t1_2mm.nii'], classes=2):
    # TODO fliter sMCI and pMCI
    label_df = pd.read_csv(label_csv_path, index_col='ID')
    subjects = []
    for center in centers:
        for subject in center.subjects:
            nii_pathes = [subject['baseline']['t1'].build_path(path) for path in img_pathes]
            for nii_path in nii_pathes:
                if not os.path.exists(nii_path):
                    print(f'No file for {center.name}:{subject.name}')

    for center in centers:
        for subject in center.subjects:
            nii_pathes = [subject['baseline']['t1'].build_path(path) for path in img_pathes]
            img_count = len(nii_pathes)
            try:
                label = label_df.loc[subject.name]['Label']

                if label == 'sMCI':
                    label = 0
                elif label == 'pMCI':
                    label = 1

                # Must contain at least one image in __init__
                tio_subject = tio.Subject(
                    img0=tio.ScalarImage(nii_pathes[0]),
                    img_count=img_count,
                    label=label,
                    path=subject.fullpath,
                    center=center.name,
                    name=subject.name
                )

                for i in range(1, img_count):
                    tio_subject.add_image(tio.ScalarImage(nii_pathes[i]), f'img{i}')

                subjects.append(tio_subject)
            except KeyError:
                pass
    return subjects

if __name__ == '__main__':
    # Load AD_NC model -> finetune for sMCI, pMCI

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--classes', '-c', type=int, default=2, required=False)
    parser.add_argument('--dataset_path', '-dp', type=str, default='/user/home/xpkang/data/AD', required=False)
    parser.add_argument('--image_type', '-it', type=str, default='G', required=False)
    parser.add_argument('--image_size', '-is', type=int, default=2, required=False)
    parser.add_argument("--train_datasets", '-trd', nargs="+", default=['ADNI'], required=False)
    parser.add_argument("--test_datasets", '-ted', nargs="+", default=None, required=False)

    parser.add_argument('--model_dir', '-md', type=str, required=True)
    
    parser.add_argument('--channels', '-ch', type=int, default=1, required=False)

    parser.add_argument('--optimizer_type', '-ot', type=str, default='Adam', required=False)
    parser.add_argument('--lr', '-lr',type=float, default=0.001, required=False)
    parser.add_argument('--epoches', '-e', type=int, default=100, required=False)
    parser.add_argument('--batch_size', '-bs', type=int, default=8, required=False)
    parser.add_argument('--num_workers', '-nw', type=int, default=4, required=False)
    parser.add_argument("--save_every_n", type=int, default=50, required=False)
    parser.add_argument("--patience", type=int, default=20, required=False)
    parser.add_argument("--train_transform", '-tt', nargs="+", default=['noise', 'blur', 'flip', 'spatial'], required=False)

    parser.add_argument("--use_age", '-ua', type=int, default=0, required=False)

    parser.add_argument('--adni_info_path', '-aip', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()

    print(torch.cuda.is_available())
    dtime = datetime.datetime.now()
    dtime = str(dtime).replace('.', '').replace(':', '').replace(' ', '')
    _dir = os.path.join('./result', dtime)
    os.mkdir(_dir)

    db_setting = presets.get_db_setting(args.classes, args.dataset_path,
                                        args.image_type, args.image_size,
                                        args.train_datasets, args.test_datasets)

    train_setting = presets.get_train_setting(args.optimizer_type, args.lr,
                                            args.epoches, args.batch_size,
                                            args.num_workers, args.save_every_n, 
                                            args.patience, args.train_transform)

    # DB_GWM2: model setting-> channel=2
    db_setting.subject_type = 'spMCI'

    with open(os.path.join(_dir,'db_setting.pkl'), 'wb') as f:
        pickle.dump(db_setting, f)
    with open(os.path.join(_dir,'train_setting.pkl'), 'wb') as f:
        pickle.dump(train_setting, f)

    # load model from previous AD-NC model
    model_dir = args.model_dir

    with open(os.path.join(model_dir, 'model_setting.pkl'), 'rb') as f:
        model_setting = pickle.load(f)

    # find best Model
    best_acc = 0
    best_model_path = None
    files = os.listdir(model_dir)
    for f in files:
        if 'Model' in f and '.pt' in f:
            acc_loc = f.find('acc')
            acc = float(f[acc_loc+3:acc_loc+7])
            print(acc)
            if acc > best_acc:
                best_model_path = f
                best_acc = acc

    full_model_path = os.path.join(model_dir, best_model_path)
    print(f'Model path:{full_model_path}')

    with open(os.path.join(_dir, 'log.txt'), 'w') as f:
        print('Work started at: {}'.format(str(dtime)), file=f)
        print(db_setting, file=f)

        print('-----------------------------------------------', file=f)
        print('Start Loading Datasets (sMCI and pMCI, only ADNI)', file=f)
        print(f'Load Model path:{full_model_path}', file=f)
        print(f'Timestamp: {time.time()}', file=f)

        centers = []
        for train_path in db_setting.train_pathes:
            centers += datasets.load_dataset(train_path)

        subjects = load_subjects(centers, args.adni_info_path, db_setting.img_path, classes=db_setting.classes)

        random.shuffle(subjects)

        with open(os.path.join(_dir,'subjects.pkl'), 'wb') as ff:
            pickle.dump(subjects, ff)

        for subject in tqdm(subjects):
            subject.load()

        print('Finish Loading Datasets', file=f)
        print(f'Timestamp: {time.time()}', file=f)

        train_transform = data_argument.build_transform(train_setting.train_transform, shape=model_setting.image_sizes)
        test_transform = data_argument.build_transform(train_setting.test_transform, shape=model_setting.image_sizes)

        print(train_setting, file=f)

        print('-----------------------------------------------', file=f)
        print('Start Training', file=f)
        print(f'Timestamp: {time.time()}', file=f)

        device = torch.device("cuda:0")

        test_accs = []
        best_valid_accs = []
        
        kf = KFold(n_splits=10)

        print('-----------------------------------------------', file=f)
        print('Start Training', file=f)
        print(f'Timestamp: {time.time()}', file=f)
        for fold, (train_index, test_index) in enumerate(kf.split(subjects)):
            fold_dir = os.path.join(_dir, f'fold{fold}')
            if not os.path.exists(fold_dir):
                os.mkdir(fold_dir)

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

            # load from AD-NC best model every fold
            model.load_state_dict(torch.load(full_model_path, map_location=device))

            if fold == 0:
                print(model, file=f)
                summary_shape = (train_setting.batch_size, model_setting.channels,
                                model_setting.image_sizes[0], 
                                model_setting.image_sizes[1], 
                                model_setting.image_sizes[2])
                if not args.use_age:
                    print(torchinfo.summary(model, [summary_shape]), file=f)
                else:
                    print(torchinfo.summary(model, [summary_shape, (train_setting.batch_size, 2)]), file=f)

            if train_setting.optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=train_setting.lr)

            # useless
            scheduler = MultiStepLR(optimizer, milestones=[300,400], gamma=0.1)

            criterion = nn.CrossEntropyLoss().cuda()

            with open(os.path.join(fold_dir, f'train.pkl'), 'wb') as ff:
                pickle.dump(train_index, ff)
            with open(os.path.join(fold_dir, f'test.pkl'), 'wb') as ff:
                pickle.dump(test_index, ff)

            total_train_subjects = np.array(subjects)[train_index].tolist()
            train_subjects, valid_subjects = train_test_split(total_train_subjects, test_size=0.05)
            test_subjects = np.array(subjects)[test_index].tolist()

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
            lowest_loss = 10000

            train_losses = []
            train_accs = []
            valid_losses = []
            valid_accs = []

            for epoch in range(train_setting.epoches):
                if not args.use_age:
                    train_loss, train_acc = model_utils.train_model(model, train_loader, optimizer, scheduler, criterion, epoch, f=f)

                    valid_loss, valid_acc, *_ = model_utils.test_model(model, valid_loader, criterion, phase='valid', f=f)
                else:
                    train_loss, train_acc = model_utils.train_model2(model, train_loader, optimizer, scheduler, criterion, epoch, f=f)

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
                    torch.save(model.state_dict(), os.path.join(fold_dir, f'Model_acc{acc:.2f}_epoch{epoch}.pt'))

            if not args.use_age:
                _, acc, *_ = model_utils.test_model(model, test_loader, criterion, phase='test', f=f)
            else:
                _, acc, *_ = model_utils.test_model2(model, test_loader, criterion, phase='test', f=f)

            test_accs.append(acc)
            best_valid_accs.append(np.max(valid_accs))

            torch.save(model.state_dict(), os.path.join(fold_dir, f'Model_acc{acc:.2f}.pt'))

            df = pd.DataFrame(list(zip(train_losses, train_accs, valid_losses, valid_accs)),
                              columns =['Train_Loss', 'Train_Acc', 'Valid_Loss', 'Valid_Acc'])
            df.to_csv(os.path.join(fold_dir, f'Train_Report.csv'))

        print(f'Test_accs:{test_accs}', file=f)
        print(f'Best_Valid_accs:{best_valid_accs}', file=f)