import os
import copy
import json

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import progressbar as pb
import pandas as pd
import argparse

from datasets import SlidePatchData, OneEveryPatientSampler
from networks import SurvivalPatchCNN, NegativeLogLikelihood, SvmLoss
import metrics as mm


class NoneScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


def predict(model, dataloader, device=torch.device('cuda:0'), bar=True):
    ori_phase = model.training
    ori_device = next(model.parameters()).device
    model.eval()
    model.to(device)
    with torch.no_grad():
        if bar:
            dataloader = pb.progressbar(dataloader, prefix='Predict: ')
        preds = []
        patient_ids = []
        file_names = []
        for batch in dataloader:
            imgs = batch[0].to(device)
            patient_id, file_name = batch[-1]
            pred = model(imgs)

            preds.append(pred)
            patient_ids += list(patient_id)
            file_names += list(file_name)
        preds = torch.cat(preds, dim=0).cpu().numpy()
    res_df = pd.DataFrame({
        'score': preds, 'patient_id': patient_ids, 'file_name': file_names})
    model.train(ori_phase)
    model.to(ori_device)
    return res_df


def evaluate(
    model, dataloader, criterion, metrics,
    device=torch.device('cuda:0'), bar=True
):
    ori_phase = model.training
    ori_device = next(model.parameters()).device
    model.eval()
    model.to(device)

    history = {}
    for m in metrics:
        m.reset()
    if bar:
        dataloader = pb.progressbar(dataloader, prefix='Test: ')
    for batch_x, batch_y, (batch_ids, batch_files) in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            for m in metrics:
                if isinstance(m, mm.Loss):
                    m.add(loss.cpu().item(), batch_x.size(0))
                else:
                    m.add(scores.squeeze(), batch_y, batch_ids)
    for m in metrics:
        history[m.__class__.__name__] = m.value()
    print(
        "Test results: " +
        ", ".join([
            '%s: %.4f' % (m.__class__.__name__, history[m.__class__.__name__])
            for m in metrics
        ])
    )
    model.train(ori_phase)
    model.to(ori_device)
    return history


def train(
    model, criterion, optimizer, dataloaders, scheduler=NoneScheduler(None),
    epoch=100, device=torch.device('cuda:0'), l2=0.0,
    metrics=(mm.Loss(), mm.CIndexForSlide()), standard_metric_index=1,
    clip_grad=False
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    best_metric_name = metrics[standard_metric_index].__class__.__name__ + \
        '_valid'
    history = {
        m.__class__.__name__+p: []
        for p in ['_train', '_valid']
        for m in metrics
    }
    model.to(device)

    for e in range(epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
                prefix = "Train: "
            else:
                model.eval()
                prefix = "Valid: "
            # progressbar
            format_custom_text = pb.FormatCustomText(
                'Loss: %(loss).4f', dict(loss=0.))
            widgets = [
                prefix, " ",
                pb.Counter(),
                ' ', pb.Bar(),
                ' ', pb.Timer(),
                ' ', pb.AdaptiveETA(),
                ' ', format_custom_text
            ]
            iterator = pb.progressbar(dataloaders[phase], widgets=widgets)

            for m in metrics:
                m.reset()
            for batch_x, batch_y, (batch_ids, batch_files) in iterator:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logit = model(batch_x)
                    loss = criterion(logit, batch_y)
                    # 只给weight加l2正则化
                    if l2 > 0.0:
                        for p_n, p_v in model.named_parameters():
                            if p_n == 'weight':
                                loss += l2 * p_v.norm()
                    if phase == 'train':
                        loss.backward()
                        if clip_grad:
                            nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1)
                        optimizer.step()
                with torch.no_grad():
                    for m in metrics:
                        if isinstance(m, mm.Loss):
                            m.add(loss.cpu().item(), batch_x.size(0))
                            format_custom_text.update_mapping(loss=m.value())
                        else:
                            m.add(logit.squeeze(), batch_y, batch_ids)

            for m in metrics:
                history[m.__class__.__name__+'_'+phase].append(m.value())
            print(
                "Epoch: %d, Phase:%s, " % (e, phase) +
                ", ".join([
                    '%s: %.4f' % (
                        m.__class__.__name__,
                        history[m.__class__.__name__+'_'+phase][-1]
                    ) for m in metrics
                ])
            )

            if phase == 'valid':
                epoch_metric = history[best_metric_name][-1]
                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

    print("Best metric: %.4f" % best_metric)
    model.load_state_dict(best_model_wts)
    return model, history


def check_update_dirname(dirname):
    if os.path.exists(dirname):
        dirname += '-'
        check_update_dirname(dirname)
    else:
        os.makedirs(dirname)
        return dirname


def main():

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--save', default='./save',
        help='保存的文件夹路径，如果有重名，会在其后加-来区别'
    )
    parser.add_argument(
        '-is', '--image_size', default=224, type=int,
        help='patch会被resize到多大，默认时224 x 224'
    )
    parser.add_argument(
        '-ts', '--test_size', default=0.2, type=float,
        help='测试集的大小，默认时0.2'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=64, type=int,
        help='batch size，默认时64'
    )
    parser.add_argument(
        '-nw', '--num_workers', default=12, type=int,
        help='多进程数目，默认时12'
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=0.0001, type=float,
        help='学习率大小，默认时0.0001'
    )
    parser.add_argument(
        '-e', '--epoch', default=10, type=int,
        help='epoch 数量，默认是10'
    )
    parser.add_argument(
        '-tp', '--test_patches', default=2, type=int,
        help='测试时随机从每个patient中抽取的patches的数量，默认是2'
    )
    parser.add_argument(
        '--cindex_reduction', default='mean',
        help='聚合同一张slide的patches时的聚合方式，默认时mean'
    )
    parser.add_argument(
        '--loss_type', default='cox',
        help='使用的loss的类型，默认是cox，也可以是svmloss'
    )
    args = parser.parse_args()
    save = args.save
    image_size = (args.image_size, args.image_size)
    test_size = args.test_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.learning_rate
    epoch = args.epoch
    test_patches = args.test_patches
    cindex_reduction = args.cindex_reduction


    # ----- 读取数据 -----
    demographic_file = '/home/dl/NewDisk/Slides/TCGA-OV/demographic.csv'
    tiles_dir = '/home/dl/NewDisk/Slides/TCGA-OV/Tiles'

    dat = SlidePatchData.from_demographic(
        demographic_file, tiles_dir, transfer=transforms.ToTensor()
    )
    train_transfer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transfer = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dat, valid_dat = dat.split_by_patients(
        test_size, train_transfer=train_transfer, test_transfer=test_transfer)
    train_sampler = OneEveryPatientSampler(train_dat)
    test_sampler = OneEveryPatientSampler(
        valid_dat, num_per_patients=test_patches)
    dataloaders = {
        'train': data.DataLoader(
            train_dat, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers),
        'valid': data.DataLoader(
            valid_dat, batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers)
    }

    # ----- 构建网络和优化器 -----
    net = SurvivalPatchCNN()
    if args.loss_type == 'cox':
        criterion = NegativeLogLikelihood()
    elif args.loss_type == 'svmloss':
        criterion = SvmLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scorings = [mm.Loss(), mm.CIndexForSlide(reduction=cindex_reduction)]

    # ----- 训练网络 -----
    net, hist = train(
        net, criterion, optimizer, dataloaders, epoch=epoch, metrics=scorings
    )

    # 保存结果
    dirname = check_update_dirname(save)
    torch.save(net.state_dict(), os.path.join(dirname, 'model.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(dirname, 'train.csv'))
    with open(os.path.join(dirname, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
