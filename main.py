
import numpy as np
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import os
from scipy.stats import gmean


"""Train XGBoost model using feature extracted.

Input:
    make sure features are already extracted using feature.py

Output:
    stage 2 submission: subm.csv file

"""


def train_xgboost():
    df = pd.read_csv('data/stage1_labels.csv')
#    print df.head()

    x = []
    y = []
    did = df['id'].tolist()
    cancer = df['cancer'].tolist()
    for i in range(len(df)):
        if os.path.isfile('data/stage1/%s.npy' % did[i]):
            f = np.load('data/stage1/%s.npy' % did[i])
            f = f.reshape(f.shape[0], 2048)
            x.append(np.mean(f, axis=0))
            y.append(cancer[i])

    x = np.array(x)
    print x.shape
    y = np.array(y)

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=822, stratify=y, test_size=0.1)

    clfs = []
    for s in range(5):
	# Some parameters were taken from discussion.
        clf = xgb.XGBRegressor(n_estimators=1000, max_depth=10, min_child_weight=10,
                               learning_rate=0.01, subsample=0.80, colsample_bytree=0.70,
                               seed=822 + s, reg_alpha=0.1)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        clfs.append(clf)
    return clfs


def make_submission():
    clfs = train_xgboost()
    df = pd.read_csv('data/stage2_sample_submission.csv')
    x = np.array([np.mean(np.load('data/stage2/%s.npy' % str(did)).reshape(-1, 2048), axis=0)
                  for did in df['id'].tolist()])
    preds = []

    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.001, 1))

    pred = gmean(np.array(preds), axis=0)
#    print pred

    df['cancer'] = pred
    df.to_csv('subm.csv', index=False)
#    print df.head()


if __name__ == '__main__':
    make_submission()

