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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--classes', '-c', type=int, default=2, required=False)
    parser.add_argument('--dataset_path', '-dp', type=str, default='/user/home/xpkang/data/AD', required=False)
    parser.add_argument('--image_type', '-it', type=str, default='G', required=False)
    parser.add_argument('--image_size', '-is', type=int, default=2, required=False)
    parser.add_argument("--train_datasets", '-trd', nargs="+", default=['ADNI', 'MCAD', 'EDSD'], required=False)
    parser.add_argument("--test_datasets", '-ted', nargs="+", default=None, required=False)

    parser.add_argument('--model', '-m', type=str, default='NEST2_SMALL', required=False)
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
    model_setting = presets.get_model_setting(args.model)
    train_setting = presets.get_train_setting(args.optimizer_type, args.lr,
                                            args.epoches, args.batch_size,
                                            args.num_workers, args.save_every_n, 
                                            args.patience, args.train_transform)

    # DB_GWM2: model setting-> channel=2
    model_setting.channels = args.channels
    model_setting.num_classes = args.classes

    with open(os.path.join(_dir,'db_setting.pkl'), 'wb') as f:
        pickle.dump(db_setting, f)
    with open(os.path.join(_dir,'model_setting.pkl'), 'wb') as f:
        pickle.dump(model_setting, f)
    with open(os.path.join(_dir,'train_setting.pkl'), 'wb') as f:
        pickle.dump(train_setting, f)

    with open(os.path.join(_dir, 'log.txt'), 'w') as f:
        print('Work started at: {}'.format(str(dtime)), file=f)
        print(db_setting, file=f)

        print('-----------------------------------------------', file=f)
        print('Start Loading Datasets', file=f)
        print(f'Timestamp: {time.time()}', file=f)

        centers = []
        for train_path in db_setting.train_pathes:
            centers += datasets.load_dataset(train_path)

        if not args.use_age:
            subjects = dataloader.load_subjects(centers, db_setting.img_path, classes=db_setting.classes)
        else:
            subjects = dataloader2.load_subjects(centers, db_setting.img_path, classes=db_setting.classes)

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
                    torch.save(model.state_dict(), os.path.join(fold_dir, f'Model_acc{acc:.4f}_epoch{epoch}.pt'))

            if not args.use_age:
                _, acc, *_ = model_utils.test_model(model, test_loader, criterion, phase='test', f=f)
            else:
                _, acc, *_ = model_utils.test_model2(model, test_loader, criterion, phase='test', f=f)

            test_accs.append(acc)
            best_valid_accs.append(np.max(valid_accs))

            torch.save(model.state_dict(), os.path.join(fold_dir, f'Model_acc{acc:.4f}.pt'))

            df = pd.DataFrame(list(zip(train_losses, train_accs, valid_losses, valid_accs)),
                              columns =['Train_Loss', 'Train_Acc', 'Valid_Loss', 'Valid_Acc'])
            df.to_csv(os.path.join(fold_dir, f'Train_Report.csv'))

        print(f'Test_accs:{test_accs}', file=f)
        print(f'Best_Valid_accs:{best_valid_accs}', file=f)