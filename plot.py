import os
import sys
import json

import matplotlib.pyplot as plt
import pandas as pd


def test_plot():
    test_reses = []
    for i in range(10):
        res_path = './RESULTS/zoom20_rr%d/' % i
        test_file = os.path.join(res_path, 'test.json')
        with open(test_file, 'r') as f:
            test_res = json.load(f)
        test_res['rank_ratio'] = i / 10
        test_reses.append(test_res)
    test_reses = pd.DataFrame(test_reses)
    test_reses = test_reses.set_index('rank_ratio')
    test_reses['CIndexForSlide'].plot()
    plt.show()


def train_plot():
    train_reses = []
    for i in range(10):
        res_path = './RESULTS/zoom20_rr%d/' % i
        train_file = os.path.join(res_path, 'train.csv')
        train_res = pd.read_csv(train_file, index_col=0)
        train_res.index.name = 'epoch'
        # 将columns设置成multiindex
        multiindex = [tuple(c.split('_')) for c in train_res.columns]
        multiindex = pd.MultiIndex.from_tuples(
            multiindex, names=['metric', 'phase'])
        train_res.columns = multiindex
        train_res = train_res.stack(level=[0, 1]).reset_index()

        train_res['rank_ratio'] = i / 10
        train_reses.append(train_res)
    train_reses = pd.concat(train_reses)
    train_reses.rename({0: 'value'}, axis=1, inplace=True)

    all_metrics = train_reses['metric'].unique()
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, p in enumerate(['train', 'valid']):
        for j, m in enumerate(all_metrics):
            ax = axes[i, j]
            for rr in range(10):
                rr /= 10
                subdf = train_reses[
                    (train_reses['phase'] == p) &
                    (train_reses['metric'] == m) &
                    (train_reses['rank_ratio'] == rr)
                ]
                xvalue = subdf['epoch'].values
                yvalue = subdf['value'].values
                ax.plot(xvalue, yvalue, label=('rr=%.1f' % rr))
            ax.set_title('Phase: %s, Metric: %s' % (p, m))
            ax.set_xlabel('epoch')
            ax.set_ylabel(m)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == 'train':
        train_plot()
    elif arg == 'test':
        test_plot()
    else:
        raise ValueError()
