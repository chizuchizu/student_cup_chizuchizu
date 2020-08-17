import string
import pandas as pd
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torchtext


#confing
SEED = 2020
BASE_PATH = './data/'
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4



train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)
train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
train['label'] -= 1

test = pd.read_csv(BASE_PATH + "test.csv").drop(['id'], axis=1)
test = test.rename(columns={TEXT_COL: 'text', TARGET: 'label'})

train.to_csv(BASE_PATH + 'train_x.csv',index = False)
test.to_csv(BASE_PATH + 'test_x.csv',index = False)

def preprocessing_text(text):
    for p in string.punctuation:
        text = text.replace(p,'')
    text = text.strip().split()
    return text

TEXT = torchtext.data.Field(sequential=True, 
                            tokenize=preprocessing_text,
                            use_vocab = True,
                            lower = True)

LABEL = torchtext.data.Field(sequential=False,
                             use_vocab = False)

train_ds, test_ds = torchtext.data.TabularDataset.splits(
    path = BASE_PATH, train = 'train_x.csv',
    test = 'test_x.csv', format = 'csv',
    fields = [('text',TEXT), ('label', LABEL)]
)

train_dl = torchtext.data.Iterator(train_ds, batch_size = 32, train = True)
print(vars(train_ds[1]))