import numpy as np
import pandas as pd
import torch
from torchnet.meter.meter import Meter

from sksurv.metrics import concordance_index_censored


class Loss(Meter):
    def __init__(self):
        super(Loss, self).__init__()
        self.reset()

    def reset(self):
        self.running_loss = 0.
        self.num_samples = 0

    def add(self, batch_loss, batch_size):
        self.running_loss += batch_loss * batch_size
        self.num_samples += batch_size

    def value(self):
        return self.running_loss / self.num_samples


class CIndexForSlide(Meter):
    def __init__(self, hazard=True, reduction='mean'):
        super(CIndexForSlide, self).__init__()
        self.tensor = None
        self.hazard = hazard
        self.reduction = reduction
        self.reset()

    def __call__(self, output, target):
        self.reset()
        self.add(output, target)
        res = self.value()
        self.reset()
        return res

    def reset(self):
        self.hazard_scores = []
        self.status_times = []
        self.ids = []

    def add(self, hazard_score, status_time, patient_id):
        if self.tensor is None:
            self.tensor = isinstance(hazard_score, torch.Tensor)
        if self.tensor:
            hazard_score = hazard_score.detach().cpu().numpy()
            status_time = status_time.detach().cpu().numpy()
        self.hazard_scores.append(hazard_score)
        self.status_times.append(status_time)
        self.ids += list(patient_id)

    def value(self):
        status_time = np.concatenate(self.status_times, axis=0)
        self.results = pd.DataFrame({
            'hazard_score': np.concatenate(self.hazard_scores, axis=0),
            'statu': status_time[:, 0],
            'time': status_time[:, 1],
            'patient_id': self.ids
        })
        self.reduction_res = self.results.groupby(
            'patient_id').agg({
                'hazard_score': self.reduction,
                'statu': lambda x: x.iloc[0],
                'time': lambda x: x.iloc[0]
            })
        return self.func(
            self.reduction_res[['statu', 'time']].values,
            self.reduction_res['hazard_score'].values
        )

    def func(self, targets, outputs):
        status, time = targets[:, 0], targets[:, 1]
        status = status.astype('bool')
        if not self.hazard:
            y_pred = -y_pred
        return concordance_index_censored(status, time, outputs)[0]
