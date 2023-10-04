import os
import pickle

import pandas as pd
import torch
from torch import nn

import model_utils
from data import data_argument
from data import dataloader
from datamanagement import datasets
from models import nest
from models import vit

if __name__ == '__main__':
    # Readin model trained from all data
    result_dir = './result/2023-02-25ViTSG2_ADNC_ALL'

    # Readin original database and model setting
    with open(os.path.join(result_dir,'db_setting.pkl'), 'rb') as f:
        db_setting = pickle.load(f)
    with open(os.path.join(result_dir,'model_setting.pkl'), 'rb') as f:
        model_setting = pickle.load(f)
    with open(os.path.join(result_dir,'train_setting.pkl'), 'rb') as f:
        train_setting = pickle.load(f)
    
    model_path = result_dir

    # Readin all subjects
    db_setting.classes = 3

    all_names = []
    all_y_prob = [[] for _ in range(2)]
    all_y_true = []

    # test in all subjects (but we only takes the MCI for analysis cause model already seen AD and NC subjects)
    db_setting.test_datasets = ['MCAD', 'EDSD', 'ADNI']
    centers = []
    for path in db_setting.test_pathes:
        centers += datasets.load_dataset(path)

    test_subjects = dataloader.load_subjects(centers, db_setting.img_path, classes=3)

    print(len(test_subjects))
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
    if model_setting.type == 'nest':
        model = nest.NesT(model_setting).cuda()
    elif model_setting.type == 'vit':
        model = vit.Vit(model_setting).cuda()

    device = torch.device('cuda:0')

    model.load_state_dict(torch.load(full_model_path, map_location=device))

    criterion = nn.CrossEntropyLoss().cuda()
    *_, names, y_prob, y_true  = model_utils.predict(model, test_loader, phase='predict')

    y_prob = y_prob.numpy()
    y_true = y_true.numpy()
    print(y_prob.shape, y_true.shape)

    # record name, y_prob, y_true for each fold
    all_names += names
    for i in range(y_prob.shape[1]):
        all_y_prob[i] += y_prob.T[i].tolist()
    all_y_true += y_true.tolist()

    # save name, y_prob, y_true
    data = {'names': all_names, 'y_true': all_y_true}
    for i in range(y_prob.shape[1]):
        data[f'y_prob_{i}'] = all_y_prob[i]
    df = pd.DataFrame.from_dict(data)
    df.to_csv(os.path.join(result_dir, 'all_test_result.csv'))