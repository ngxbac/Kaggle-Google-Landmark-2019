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

    test_csv = '/raid/data/kaggle/landmark_recognition/new_data/recognition_sample_submission.csv'

    train_df = pd.read_csv('/raid/bac/kaggle/landmark/csv/train_popular.csv.gz', usecols=['landmark_id'])
    le = LabelEncoder()
    train_df['landmark_id'] = le.fit_transform(train_df['landmark_id'])

    model_name = 'resnet50'

    # Dataset
    dataset = LandmarkDataset(
        df=test_csv,
        root='/raid/data/kaggle/landmark_recognition/new_data/test/',
        transform=valid_aug(224),
        mode='infer'
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    model = FewShotModel(
        extractor_name=model_name,
        num_classes=92726,
        n_embedding=2048
    )

    checkpoint = "/raid/bac/kaggle/logs/landmark/resnet50/checkpoints//stage1.iter.112500.pth"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    pred = predict(model, loader)

    pred_cls = np.argmax(pred, axis=1)
    confs = []
    for i, cls in enumerate(tqdm(pred_cls, total=len(pred_cls))):
        confs.append(pred[i, cls])

    pred_original_cls = []
    for cls in tqdm(pred_cls, total=len(pred_cls)):
        pred_original_cls.append(le.inverse_transform([cls])[0])

    submission = dataset.df.copy()

    landmarks = []
    for label, conf in zip(pred_original_cls, confs):
        landmarks.append(f'{label} {conf}')

    submission['landmarks'] = landmarks
    os.makedirs('submission', exist_ok=True)
    submission.to_csv(f'./submission/{model_name}_kfold.csv', index=False)



