import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import os
from tqdm import *
from albumentations import *

"""
Define the model
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from cnn_finetune import make_model


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class FewShotModel(nn.Module):
    def __init__(self, extractor_name='se_resnext50_32x4d', n_embedding=2048, num_classes=100, norm=True, scale=True):
        super(FewShotModel, self).__init__()
        self.extractor = Extractor(extractor_name)
        self.n_features = self.extractor.n_features
        self.embedding = Embedding(self.n_features, n_embedding)
        self.classifier = Classifier(n_embedding, num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = F.normalize(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = F.normalize(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))


class Extractor(nn.Module):
    def __init__(self, extractor_name):
        super(Extractor,self).__init__()
        basenet = make_model(
            model_name=extractor_name,
            num_classes=1000,
            input_size=(224, 224),
            pretrained=True
        )
        # print(basenet)

        self.n_features = basenet._classifier.in_features
        self.pool = GeM()

        self.extractor = basenet._features
        # for param in self.extractor.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


class Embedding(nn.Module):
    def __init__(self, in_features, n_embedding):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(in_features, n_embedding)

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_embedding, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(n_embedding, num_classes, bias=None)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        self.fc.weight.data = F.normalize(self.fc.weight.data)
        x = self.fc(x)
        return x


"""
Define the dataset
"""

def load_image(path):
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image = np.zeros((224, 224, 3))

    return image


class LandmarkRetrivalDataset(Dataset):
    def __init__(self, df,
                 root,
                 transform):
        self.root = root
        self.transform = transform
        self.ids = df['id'].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        # Adjust the path of the image
        path = os.path.join(self.root, id + '.jpg')
        # Load the image
        image = load_image(path)
        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image
        }


"""
Augmentations
"""

def valid_aug(image_size=224):
    return Compose([
        Resize(256, 256),
        CenterCrop(image_size, image_size),
        Normalize()
    ], p=1)


device = torch.device('cuda')

def extract_features(model, loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            embedding = model.extract(images)
            embedding = embedding.detach().cpu().numpy()
            embeddings.append(embedding)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


if __name__ == '__main__':

    nrows = 1000 # Just for debug
    index_df = pd.read_csv('your_index_csv', nrows=nrows)
    root = 'your_root'
    model_name = 'se_resnext50_32x4d'

    # Dataset
    dataset = LandmarkRetrivalDataset(
        df=index_df,
        root=root,
        transform=valid_aug(224),
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

    # Your checkpoint path
    checkpoint = "/raid/bac/kaggle/logs/landmark/resume2/se_resnext50_32x4d/checkpoints//stage1.4.pth"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    embedding = extract_features(model, loader)
    np.save('your_embedding.npy', embedding)
