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


class SvmLoss(nn.Module):
    def __init__(self, reduction='mean', rank_ratio=1.0):
        super(SvmLoss, self).__init__()
        if reduction == 'mean':
            self.agg_func = torch.mean
        elif reduction == 'sum':
            self.agg_func = torch.sum
        self.rank_ratio = rank_ratio

    def forward(self, logit, status_time):
        status, time = status_time[:, 0].float(), status_time[:, 1].float()
        logit = -logit.squeeze()  # 加一个-使得我们得到得总是risk
        # rank loss
        low, high = self._comparable_pairs(status, time)
        low_logits, high_logits = logit[low], logit[high]
        rank_loss = self.agg_func(
            (1 - high_logits + low_logits).clamp(min=0) ** 2)
        # regress loss
        uncensor_part = time - logit
        censor_part = uncensor_part.clamp(min=0)
        reg_loss = self.agg_func(
            ((1 - status) * censor_part + status * uncensor_part) ** 2)
        # 总loss
        loss = self.rank_ratio * rank_loss + (1 - self.rank_ratio) * reg_loss
        return loss

    @staticmethod
    def _comparable_pairs(status, time):
        ''' 得到可比较的样本对，其中生存时间段的在前面 '''
        batch_size = len(status)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个生存时间小于第二个的元素
        time_mask = time[pairs1] < time[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 选择生存时间小的event必须是1
        event_mask = status[pairs1] == 1
        pairs1, pairs2 = pairs1[event_mask], pairs2[event_mask]
        return pairs1, pairs2


if __name__ == '__main__':
    test()
