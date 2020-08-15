import os, gc, sys
import random

import pandas as pd
import numpy as np

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
augmentation = True
memo = "group_kfold"
# 1セットあたりのデータ
SET_NUM = 2
params = {
    # "output_dir": "outputs/",
    "max_seq_length": 128,
    "train_batch_size": 32,
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


seed_everything(SEED)

train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)
train_aug = pd.read_csv(BASE_PATH + "train_fr_en.csv").rename(columns={"transrated": 'text', TARGET: 'label'})
train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
train['label'] -= 1
# train["text"] = train["text"].str.lower()

train_aug["label"] -= 1

length = len(train)
train.index = range(0, length * 2, 2)
train_aug.index = range(1, length * 2, 2)

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

    X_train = train.loc[train_idx]
    X_valid = train.loc[valid_idx]

    # print(weight)
    # print(type(weight))
    model = ClassificationModel(model_type=MODEL_TYPE, model_name=MODEL_NAME, num_labels=4,
                                args=params, use_cuda=True, weight=weight.tolist())

    model.train_model(X_train)

    result, model_outputs, wrong_predictions = model.eval_model(X_valid, f1=metric_f1)
    print(result)
    f1_score += result / N_FOLDS

    fold_pred, raw_outputs = model.predict(test['text'])
    y_pred[:, fold] = fold_pred
    # y_pred += fold_pred / N_FOLDS
    # print(y_pred)

# 最頻値
y_pred = stats.mode(y_pred, axis=1)[0].flatten().astype(int)

test = pd.read_csv(BASE_PATH + "test.csv")

submit = pd.DataFrame({'index': test['id'], 'pred': y_pred + 1})

aug = "using_aug" if augmentation else "non_aug"

submit.to_csv(f"../outputs/submit_{aug}_{MODEL_NAME}_{round(f1_score, 3)}_{memo}.csv", index=False, header=False)
