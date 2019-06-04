import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import glob
import click
from tqdm import *

from sklearn.preprocessing import LabelEncoder

from models import FewShotModel
from augmentation import *
from dataset import LandmarkDataset


device = torch.device('cuda')

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = F.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


if __name__ == '__main__':

    test_csv = '/raid/bac/kaggle/landmark/csv/recognition_sample_submission.csv'

    train_df = pd.read_csv('/raid/bac/kaggle/landmark/csv/train_popular.csv.gz', usecols=['landmark_id'])
    le = LabelEncoder()
    train_df['landmark_id'] = le.fit_transform(train_df['landmark_id'])
    all_class = le.classes_

    i2c = {}
    for i, cls in enumerate(all_class):
        i2c[i] = cls


    # assert le.inverse_transform([2106])[0] == i2c[2106]

    model_name = 'se_resnext50_32x4d'

    model = FewShotModel(
        extractor_name=model_name,
        num_classes=92726,
        n_embedding=2048
    )

    checkpoint = "/raid/bac/kaggle/logs/landmark/resume2/se_resnext50_32x4d/checkpoints//stage1.3.pth"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    tta_aug = [valid_aug_hflip(224), valid_aug(224)]

    pred_tta = []
    for i, tta in enumerate(tta_aug):
        # Dataset
        dataset = LandmarkDataset(
            df=test_csv,
            root='/raid/data/kaggle/landmark_recognition/new_data/test_state2_2/',
            transform=tta,
            mode='infer'
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )

        # model = model.to(device)
        # pred = predict(model, loader)
        # pred_tta.append(pred)
        # np.save(f'./submission/{model_name}_stage2_resume2_epoch4_tta_from_0_{i}.npy', pred)

        pred = np.load(f'./submission/{model_name}_stage2_resume2_epoch4_tta_from_0_{i}.npy')
        pred_tta.append(pred)
        del pred

    # import pdb
    # pdb.set_trace()
    pred = (pred_tta[0] + pred_tta[1]) / 2
    pred_tta = pred
    del pred
    # pred_tta = np.asarray(pred_tta).mean(axis=0)

    pred_cls = np.argmax(pred_tta, axis=1)
    confs = []
    for i, cls in enumerate(tqdm(pred_cls, total=len(pred_cls))):
        confs.append(pred_tta[i, cls])

    pred_original_cls = []
    for cls in tqdm(pred_cls, total=len(pred_cls)):
        pred_original_cls.append(i2c[cls])

    submission = dataset.df.copy()

    landmarks = []
    for label, conf in zip(pred_original_cls, confs):
        landmarks.append(f'{label} {conf}')

    submission['landmarks'] = landmarks
    os.makedirs('submission', exist_ok=True)
    submission.to_csv(f'./submission/{model_name}_stage2_resume2_epoch4_tta_from_0.csv', index=False)
    np.save(f'./submission/{model_name}_stage2_resume2_epoch4_tta_from_0.npy', pred_tta)



