import torch.nn.functional as F
import torch
import torch.nn as nn
from cnn_finetune import make_model
from src.cirtorch.layers.pooling import GeM


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
        x = self.l2_norm(x)
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
