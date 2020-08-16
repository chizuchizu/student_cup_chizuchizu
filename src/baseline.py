import os, gc, sys
import random

import pandas as pd
import numpy as np
import pulp

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from scipy import stats

from simpletransformers.classification import ClassificationModel
import torch

SEED = 2020
BASE_PATH = '../data/'
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
MODEL_TYPE = "bert"
MODEL_NAME = "bert-base-uncased"
augmentation = False
memo = "hack_code_"
# 1セットあたりのデータ
SET_NUM = 2
params = {
    # "output_dir": "outputs/",
    "max_seq_length": 64,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "reprocess_input_data": True,
    "do_lower_case": True,
    "manual_seed": SEED,
    "verbose": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
}


def metric_f1(labels, preds):
    from sklearn.metrics import f1_score
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


N_CLASSES = [404, 320, 345, 674]  # @yCarbonによる推定（過去フォーラム参照）


# 制約付き対数尤度最大化問題を解く
def hack(prob):
    prob = np.where(prob < 0, 0, prob)
    logp = np.log(prob + 1e-4)
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


seed_everything(SEED)

train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)
train_aug = pd.read_csv(BASE_PATH + "train_fr_en.csv").rename(columns={"transrated": 'text', TARGET: 'label'})
train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
train['label'] -= 1

train["text"] = train["text"].str.replace(".", "").str.strip()
# train = train[~train["text"].duplicated()]
# train["text"] = train["text"].str.lower()

train_aug["label"] -= 1

length = len(train)
train.index = range(0, length * 2, 2)
# train_aug.index = range(1, length * 2, 2)

# groups = [i for _ in range(SET_NUM) for i in range(train.shape[0])]
weight = len(train) / train["label"].value_counts().sort_index().values

if augmentation:
    train = pd.concat([train, train_aug])
    train = train.sort_index()

test = pd.read_csv(BASE_PATH + "test.csv")
test = test.rename(columns={TEXT_COL: 'text'}).drop(['id'], axis=1)

# kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
# train['fold_id'] = -1
groups = [i // SET_NUM for i in range(train.shape[0])]

y_pred = np.zeros((test.shape[0], N_FOLDS))

# print(groups)
group_kfold = GroupKFold(n_splits=N_FOLDS)
f1_score: int = 0

for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(train.index, train['label'], groups)):
    # train.loc[train.iloc[valid_idx].index, 'fold_id'] = fold

    X_train = train.iloc[train_idx]
    X_valid = train.iloc[valid_idx]

    # print(weight)
    # print(type(weight))
    model = ClassificationModel(model_type=MODEL_TYPE, model_name=MODEL_NAME, num_labels=4,
                                args=params, use_cuda=True, weight=weight.tolist())

    model.train_model(X_train)

    result, model_outputs, wrong_predictions = model.eval_model(X_valid, f1=metric_f1)
    print(result)
    f1_score += result["f1"] / N_FOLDS

    fold_pred, raw_outputs = model.predict(test['text'])

    y_pred[:, fold] = hack(raw_outputs)
    # y_pred += fold_pred / N_FOLDS
    # print(y_pred)

print(f1_score)
# 最頻値
y_pred = stats.mode(y_pred, axis=1)[0].flatten().astype(int)

test = pd.read_csv(BASE_PATH + "test.csv")

submit = pd.DataFrame({'index': test['id'], 'pred': y_pred + 1})

aug = "using_aug" if augmentation else "non_aug"

submit.to_csv(f"../outputs/submit_{aug}_{MODEL_NAME}_{round(f1_score, 3)}_{memo}.csv", index=False, header=False)
