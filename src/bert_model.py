"""
目的：パッケージ化
"""

import os
import gc
import sys
import random

import pandas as pd
import numpy as np
import pulp

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from scipy import stats

from simpletransformers.classification import ClassificationModel
import torch

N_CLASSES = [404, 320, 345, 674]


def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')


def seed_everything(seed):
    """for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 制約付き対数尤度最大化問題を解く
def hack(prob, hack=True):
    if hack:
        scaler = MinMaxScaler()
        prob = scaler.fit_transform(prob)
        print(prob)

        # prob = np.where(prob < 0, 0, prob)
        logp = np.log(prob + 1e-8)
        N = prob.shape[0]
        K = prob.shape[1]

        m = pulp.LpProblem('Problem', pulp.LpMaximize)  # 最大化問題

        # 最適化する変数(= 提出ラベル)
        x = pulp.LpVariable.dicts('x', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)

        # log likelihood(目的関数)
        log_likelihood = pulp.lpSum([x[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
        m += log_likelihood

        # 各データについて，1クラスだけを予測ラベルとする制約
        for i in range(N):
            m += pulp.lpSum([x[(i, k)] for k in range(K)]) == 1  # i.e., SOS1

        # 各クラスについて，推定個数の合計に関する制約
        for k in range(K):
            m += pulp.lpSum([x[(i, k)] for i in range(N)]) == N_CLASSES[k]

        m.solve()  # 解く

        assert m.status == 1  # assert 最適 <=>（実行可能解が見つからないとエラー）

        x_ast = np.array([[int(x[(i, j)].value()) for j in range(K)] for i in range(N)])  # 結果の取得
        return x_ast.argmax(axis=1)  # 結果をonehotから -> {0, 1, 2, 3}のラベルに変換
    else:
        return prob.argmax(axis=1)


def all_train(train, test, params, model_name, model_type, lb_hack):
    weight = len(train) / train["label"].value_counts().sort_index().values

    model = ClassificationModel(model_type=model_type, model_name=model_name, num_labels=4,
                                args=params, use_cuda=True, weight=weight.tolist())
    model.train_model(train)

    pred, raw_outputs = model.predict(test["description"])

    y_pred = hack(raw_outputs, lb_hack)

    pseudo_idx = (pd.DataFrame(y_pred).max(axis=1) > 2.5)

    return y_pred, pseudo_idx


def cross_pseudo_labeling(train, test, params, n_folds, model_name, model_type, lb_hack):
    splits = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1234).split(train["text"], train["label"])
    )
    splits_test = list(
        KFold(n_splits=n_folds, shuffle=True, random_state=1234).split(test["text"])
    )

    y_pred = np.zeros((test.shape[0], n_folds))
    oof = np.zeros(train.shape[0])
    oof_raw = np.zeros((train.shape[0], n_folds))
    weight = len(train) / train["label"].value_counts().sort_index().values

    f1_score = 0

    for fold, ((train_idx, valid_idx), (train_test_idx, test_idx)) in enumerate(zip(splits, splits_test)):
        X_train = pd.concat([train.iloc[train_idx], test.iloc[train_test_idx]])
        X_valid = train.iloc[valid_idx]
        model = ClassificationModel(model_type=model_type, model_name=model_name, num_labels=4,
                                    args=params, use_cuda=True, weight=weight.tolist())

        model.train_model(X_train)

        result, model_outputs, wrong_predictions = model.eval_model(X_valid, f1=metric_f1)
        print(result)
        f1_score += result["f1"] / n_folds

        fold_pred, raw_outputs = model.predict(test.iloc[test_idx]["text"].values)
        # y_pred[:, fold] = hack(raw_outputs)
        y_pred[test_idx, :] = raw_outputs

        oof_pred, oof_outputs = model.predict(X_valid["text"].values)  # 謎のバグが発生するので
        oof[valid_idx] = oof_pred
        oof_raw[valid_idx, :] = oof_outputs
        # oof[valid_idx] = hack(oof_outputs)

    print(f"mean f1_score: {f1_score}")

    raw_pred = y_pred.copy()

    y_pred = hack(y_pred, lb_hack)

    # oof = hack(oof_raw)

    # y_pred = stats.mode(y_pred, axis=1)[0].flatten().astype(int)

    test_pred = pd.DataFrame(np.concatenate([y_pred.reshape(-1, 1), raw_pred], 1))
    oof_pred = pd.DataFrame(np.concatenate([oof.reshape(-1, 1), oof_raw], 1))

    return test_pred, f1_score, oof_pred

def model(train, test, params, n_folds, model_name, model_type, lb_hack):
    kfold = StratifiedKFold(n_splits=n_folds)

    y_pred = np.zeros((test.shape[0], n_folds))
    oof = np.zeros(train.shape[0])
    oof_raw = np.zeros((train.shape[0], n_folds))
    weight = len(train) / train["label"].value_counts().sort_index().values

    f1_score = 0

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train["text"], train['label'])):
        X_train = train.iloc[train_idx]
        X_valid = train.iloc[valid_idx]
        model = ClassificationModel(model_type=model_type, model_name=model_name, num_labels=4,
                                    args=params, use_cuda=True, weight=weight.tolist())

        model.train_model(X_train)

        result, model_outputs, wrong_predictions = model.eval_model(X_valid, f1=metric_f1)
        print(result)
        f1_score += result["f1"] / n_folds

        fold_pred, raw_outputs = model.predict(test['description'])
        # y_pred[:, fold] = hack(raw_outputs)
        y_pred += raw_outputs / n_folds

        oof_pred, oof_outputs = model.predict(X_valid["text"].values)  # 謎のバグが発生するので
        oof[valid_idx] = oof_pred
        oof_raw[valid_idx, :] = oof_outputs
        # oof[valid_idx] = hack(oof_outputs)

    print(f"mean f1_score: {f1_score}")

    raw_pred = y_pred.copy()
    y_pred = hack(y_pred, lb_hack)

    # oof = hack(oof_raw)

    # y_pred = stats.mode(y_pred, axis=1)[0].flatten().astype(int)

    test_pred = pd.DataFrame(np.concatenate([y_pred.reshape(-1, 1), raw_pred], 1))
    oof_pred = pd.DataFrame(np.concatenate([oof.reshape(-1, 1), oof_raw], 1))

    return test_pred, f1_score, oof_pred
