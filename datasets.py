import os
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class SlidePatchData(data.Dataset):
    ''' 用于储存slide patches的pytorch datasets '''
    def __init__(self, df, transfer=None):
        '''
        df：dataframe，必须有以下5列：
            patient_id: 病人的id号，比如TCGA-23-2001;；
            patch_file: 每张patch的完整路径；
            status: 事件，0 or 1，生存分析标签;
            suvivial_time: 生存事件，生存分析标签;
            file_name: patch图像的文件名
        transfer：tranform对象，实际上就是一个callable对象，其对每一张patch进行
            操作；
        '''
        assert len(
            set(df.columns).intersection([
                'patient_id', 'patch_file', 'status',
                'survival_time', 'file_name'
            ])
        ) == 5
        self.df = df.dropna()
        self.transfer = transfer
        self._patch_counts = self.df['patient_id'].value_counts()

    def __len__(self):
        ''' 所有的patch数量 '''
        return len(self.df)

    def __getitem__(self, indx):
        '''
        根据数字指标来得到一个样本，返回的是3元组，(image, y标签, patient_id)，
        其中y标签是shape为(2,)的ndarray，其第一个元素是status、第二个元素是time
        '''
        img = Image.open(self.df['patch_file'].iloc[indx])
        if self.transfer is not None:
            img = self.transfer(img)
        y = self.df[['status', 'survival_time']].iloc[indx].values
        patient_id = self.df['patient_id'].iloc[indx]
        file_name = self.df['file_name'].iloc[indx]
        return img, y, (patient_id, file_name)

    @property
    def patients_num(self):
        ''' 数据集中病人的数量 '''
        return len(self._patch_counts)

    @property
    def patch_counts(self):
        ''' 数据集中每个病人的patch数量，是Series'''
        return self._patch_counts

    @staticmethod
    def from_demographic(
        demographic_file, tiles_dir, transfer=None, zoom='40.0', imgtype='png'
    ):
        '''
        使用整理好的demographic数据和制定的储存有patch的文件夹来建议datasets，
        其中zoom表示使用不同放大倍数的patch，其是每个病人所属文件夹的下一级文件
        夹
        '''
        demographic_df = pd.read_csv(demographic_file, index_col=False)
        demographic_use = demographic_df[[
            'patient_id', 'survival_time', 'status']]

        patient_id = []
        patient_dir = []
        for d in os.listdir(tiles_dir):
            if os.path.isdir(os.path.join(tiles_dir, d)):
                patient_id.append(d[:12])
                patient_dir.append(os.path.join(tiles_dir, d, zoom))
        patients_df = pd.DataFrame({
            'patient_id': patient_id, 'patient_dir': patient_dir})
        # 使用的是demographic数据和patch文件夹下共有的病人
        data_df = demographic_use.merge(
            patients_df, how='inner', on='patient_id')
        # 循环得到每个病人所拥有的所有patch
        patient_dir = []
        patch_files = []
        file_names = []
        for patient in data_df['patient_dir'].values:
            if os.path.exists(patient):
                for d in os.listdir(patient):
                    if d.endswith(imgtype):
                        file_names.append(d)
                        patch_files.append(os.path.join(patient, d))
                        patient_dir.append(patient)
        patch_df = pd.DataFrame({
            'patient_dir': patient_dir, 'patch_file': patch_files,
            'file_name': file_names
        })
        # 和之前的信息结合在一起
        data_df = data_df.merge(patch_df, how='inner', on='patient_dir')

        return SlidePatchData(data_df, transfer=transfer)

    def split_by_patients(
        self, test_size, seed=1234, train_transfer=None, test_transfer=None
    ):
        unique_df = self.df[['patient_id', 'status']].drop_duplicates()
        train_id, test_id = train_test_split(
            unique_df['patient_id'].values, test_size=test_size,
            stratify=unique_df['status'].values, shuffle=True,
            random_state=seed
        )

        train_df = self.df[self.df['patient_id'].isin(train_id)]
        test_df = self.df[self.df['patient_id'].isin(test_id)]

        if train_transfer is None:
            train_transfer = self.transfer

        return (
            SlidePatchData(train_df, train_transfer),
            SlidePatchData(test_df, test_transfer)
        )


def add_sequence_index(df, name='index', copy=True):
    ''' 为df增加一列，名为name，其值是range(len(df)) '''
    indx_col = list(range(len(df)))
    indx_col = np.random.permutation(indx_col)
    if copy:
        new_df = deepcopy(df)
        new_df[name] = indx_col
        return new_df
    else:
        df[name] = indx_col


def undersampling(df, min_samples, name):
    ''' 只保留df中name那一列的值不超过min_samples的样本 '''
    mask = df[name] < min_samples  # 0-365才是一共366个样本
    return df[mask]


class OneEveryPatientSampler(data.Sampler):
    '''
    自定义的sampler，用于将patches进行排序，这样得到的顺序是依次随机取出每个
    病人的一个patch，之后循环这个过程，直到某一个病人的所有patch都被取完，即
    其隐含欠采样，这只在train时使用，如果在test阶段不适用次sampler即可,
    num_per_patients控制每个人被随机出几个patch使用，默认None即使用最少patch数
    '''
    def __init__(self, dataset, random_seed=None, num_per_patients=None):
        # 设置随机数
        if random_seed is not None and random_seed is not True:
            np.random.seed(random_seed)
        if num_per_patients is None:
            # 得到slide中最小的patch数
            self.undersampling_samples = dataset.patch_counts.min()
        else:
            self.undersampling_samples = num_per_patients
        # 每个epoch的patch数量
        self._patch_nums = len(dataset.patch_counts) * \
            self.undersampling_samples

        # 只使用其中的patient id即可
        self.patient_index = dataset.df[['patient_id']]
        # 加上一列index，__iter__返回的就是index
        add_sequence_index(self.patient_index, copy=False)
        # 提前进行分组操作
        self.patient_index_group = self.patient_index.groupby(
            'patient_id', as_index=False)
        # 将分组使用的apply函数准备好
        self.apply_func = lambda df: undersampling(
            add_sequence_index(df, 'patch_index'),
            self.undersampling_samples, 'patch_index'
        )

    def __iter__(self):
        # 将随机过程放在这里，则在每次对dataloader进行for循环的时候都会产生
        # 新的随机序列
        self.patient_index = self.patient_index_group.apply(self.apply_func)
        self.patient_index = self.patient_index.sort_values(['patch_index'])
        return iter(list(self.patient_index['index'].values))

    def __len__(self):
        return self._patch_nums


def test():
    import torchvision.transforms as transforms


    demographic_file = '/home/dl/NewDisk/Slides/TCGA-OV/demographic.csv'
    tiles_dir = '/home/dl/NewDisk/Slides/TCGA-OV/Tiles'

    dat = SlidePatchData.from_demographic(
        demographic_file, tiles_dir, transfer=transforms.ToTensor()
    )

    # img, y, patient_id = dat[0]
    # img.show()
    # print(y)
    # print(patient_id)
    # print(dat.patients_num)
    # print(dat.patch_counts)
    # print(len(dat))

    sampler = OneEveryPatientSampler(dat, random_seed=True)
    # print(len(sampler))
    # print(sampler.undersampling_samples)
    # print('====================')
    # iterator = iter(sampler)
    # for i in range(5):
    #     print(next(iterator))
    # print('====================')
    # iterator = iter(sampler)
    # for i in range(5):
    #     print(next(iterator))

    dataloader = data.DataLoader(
        dat, batch_size=2, shuffle=False, sampler=sampler)
    for img, y, pid in dataloader:
        print(img[:, 0, :2, :2])
        print(pid)
        break

    for img, y, pid in dataloader:
        print(img[:, 0, :2, :2])
        print(pid)
        break

    # train_dat, test_dat = dat.split_by_patients(0.2)
    # print('=====train====')
    # print(len(train_dat))
    # print(train_dat.patients_num)
    # print('=====test====')
    # print(len(test_dat))
    # print(test_dat.patients_num)



if __name__ == '__main__':
    test()
