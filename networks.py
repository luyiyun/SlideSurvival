import torch
import torch.nn as nn
import torchvision.models as models


class SurvivalPatchCNN(nn.Module):
    def __init__(self):
        super(SurvivalPatchCNN, self).__init__()
        self.features = list(models.mobilenet_v2(True).children())[0]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=1)
        )
        # self.features = nn.Sequential(
        #     *list(models.resnet50(True).children())[:-1])
        # self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        # x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class NegativeLogLikelihood(nn.Module):
    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, logit, status_time):
        status, time = status_time[:, 0], status_time[:, 1]
        logit = logit.squeeze()
        index = time.argsort(descending=True)
        logit, status = logit[index], status[index]
        log_risk_delta = logit - logit.exp().cumsum(0).log()
        censored_risk = log_risk_delta * status.float()
        if self.reduction == 'sum':
            return -censored_risk.sum()
        return -censored_risk.mean()


def test():
    imgs = torch.rand(2, 3, 224, 224)
    net = SurvivalPatchCNN()
    out = net(imgs)
    print(out)


if __name__ == '__main__':
    test()
