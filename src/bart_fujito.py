import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from functools import partial
from torchtext.data import Dataset
from transformers import BartTokenizer, BartModel,BertModel,BertTokenizer


#config
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user', type=str,help='user name', required=True)
config = parser.parse_args()
user = config.user
if user == 'chizuchizu':
    BASE_PATH ='../data/'
else:
    BASE_PATH = './data/'

SEED = 2021
random.seed(SEED)
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
BS = 256
NUM_EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#load data 
train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)
# train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
# train['label'] -= 1

# test = pd.read_csv(BASE_PATH + "test.csv").drop(['id'], axis=1)
# test = test.rename(columns={TEXT_COL: 'text', TARGET: 'label'})


#preprocessing(以下pp)
def pp_text(text):
    text = text.lower()#小文字にする必要があるか要確認
    tokenizer = BertTokenizer.from_pretrained('facebook/bart-large')
    text = tokenizer.tokenize(text)
    return text

#dataset
TEXT = torchtext.data.Field(sequential=True,
                            tokenize=pp_text,
                            use_vocab=False,
                            batch_first=True,
                            fix_length=64,
                            init_token="[CLS]",
                            include_lengths= True,
                            lower=True)

LABEL = torchtext.data.Field(sequential=False,
                             use_vocab=False)
 
#make bart
class BART(nn.Module):
    def __init__(self):
        super(BART, self).__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-large')
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(in_features = 1024, out_features= 4)

    def forward(self, x):
        out = self.bart(x)
        out = self.dropout(out)
        out = self.cls(out)
        
        return out
#make eval
def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')
#training and eval

def eval_model(model, dl_dict, is_train = False):
    all_labels = []
    all_preds = []
    for batch in dl_dict:
        inputs = batch.text.to(device)
        if is_train:
            labels = batch.label.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            if is_train:
                all_labels += labels.tolist()
            all_preds += pred.tolist()

    return all_labels, all_preds


def train_model(model, dl_dict, loss, optim, scheduler, num_epochs):
    model.to(device)
    all_test_preds = []
    for epoch in range(num_epochs):
        all_loss = 0
        all_labels = []
        all_preds = []

        for batch in (dl_dict['train']):
            inputs = batch.text.to(device)
            label = batch.label.to(device)
            optim.zero_grad()
            all_loss += loss.item()

        print(" train | epoch ", epoch + 1, " | ", " loss ", all_loss / len(dl_dict["train"]))
        all_labels, all_preds = eval_model(model, dl_dict["val"], is_train=True)
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        print("val | epoch", epoch + 1, " | ", "f1", train_f1)
        all_test_preds.append(eval_model(model, dl_dict["test"], is_train=False)[1])

    checkpoint_weights = np.array([2 ** epoch for epoch in range(num_epochs)])
    checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()

    test_y = np.average(all_test_preds, weights=checkpoint_weights, axis=0).astype(float)
    test_y = np.round(test_y).astype(int)

    return model, test_y

class ParamScheduler:

    def __init__(self, optimizer, scale_fn, step_size):
        if not isinstance(optimizer, optim):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.scale_fn = scale_fn
        self.step_size = step_size
        self.last_batch_iteration = 0

    def batch_step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scale_fn(self.last_batch_iteration / self.step_size)

        self.last_batch_iteration += 1


def combine_scale_functions(scale_fns, phases=None):
    if phases is None:
        phases = [1. / len(scale_fns)] * len(scale_fns)
    phases = [phase / sum(phases) for phase in phases]
    phases = torch.tensor([0] + phases)
    phases = torch.cumsum(phases, 0)

    def _inner(x):
        idx = (x >= phases).nonzero().max()
        actual_x = (x - phases[idx]) / (phases[idx + 1] - phases[idx])
        return scale_fns[idx](actual_x)

    return _inner


def scale_cos(start, end, x):
    return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2
#pred
train_ds, _ = torchtext.data.TabularDataset.splits(
    path=BASE_PATH, train='train_x.csv',
    test='test_x.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)
test_ds = torchtext.data.TabularDataset(
    path=BASE_PATH + "test_x.csv",
    format="csv",
    fields=[("text", TEXT)],
)

#model instance
model = BART()

# optim
# optim = optim.Adam([
#     {'params': model.bart.encoder.layer[-1].parameters(), 'lr': 1e-4},
#     {'params': model.cls.parameters(), 'lr': 1e-4}
# ])

optim = optim.Adam(model.parameters(), lr=0.01)
# 損失関数
weight = len(train) / train["jobflag"].value_counts().sort_index().values
weights = torch.tensor(weight.tolist()).to(device)
loss = nn.CrossEntropyLoss(weight=weights)
scale_fn = combine_scale_functions(
        [partial(scale_cos, 1e-4, 5e-3), partial(scale_cos, 5e-3, 1e-3)], [0.2, 0.8])
# scheduler = ParamScheduler(optim, scale_fn, NUM_EPOCHS * len(dl_dict["train"]))
kf = KFold(n_splits=4, shuffle=True, random_state=SEED)


for tdx, vdx in kf.split(train_ds.examples):
    data_arr = np.array(train_ds.examples)

    train_dl = torchtext.data.Iterator(Dataset(data_arr[tdx], fields=[("text", TEXT), ("label", LABEL)]), batch_size=BS,
                                       train=True)
    val_dl = torchtext.data.Iterator(Dataset(data_arr[vdx], fields=[("text", TEXT), ("label", LABEL)]), batch_size=BS,
                                     train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=BS, train=False, sort=False)
    dl_dict = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    
    # scheduler = ParamScheduler(optim, scale_fn, NUM_EPOCHS * len(dl_dict["train"]))
    model, test_y = train_model(model, dl_dict, loss, optim, None ,NUM_EPOCHS)

print("DONE")

# print(model)