import string
import random
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors
from torchtext.data import Dataset
import numpy as np
from torch.optim.optimizer import Optimizer
from functools import partial

# confing
SEED = 2021
random.seed(SEED)
user = "cf"
if user == "chizuchizu":
    BASE_PATH = '../data/'
else:
    BASE_PATH = "./data/"
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
BS = 128
NUM_EPOCHS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
train['label'] -= 1

test = pd.read_csv(BASE_PATH + "test.csv").drop(['id'], axis=1)
test = test.rename(columns={TEXT_COL: 'text', TARGET: 'label'})

train.to_csv(BASE_PATH + 'train_x.csv', index=False, header=False)
test.to_csv(BASE_PATH + 'test_x.csv', index=False, header=False)
# print(train)


def preprocessing_text(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    text = text.strip().split()
    return text


TEXT = torchtext.data.Field(sequential=True,
                            tokenize="spacy",
                            use_vocab=True,
                            batch_first=True,
                            include_lengths=True,
                            fix_length=128,
                            lower=True)

LABEL = torchtext.data.Field(sequential=False,
                             use_vocab=False)

# print(vars(train_ds[0]))

# 一回
# first = True
# if not os.path.isfile(BASE_PATH + "src/.vector_cache/wiki.en.vec"):
#     fasttext = torchtext.vocab.FastText(language="en")  # 分かち書きをvecotr化するここをfnとかにしたらフランス語に対応できるかも？
# else:
fasttext = Vectors(name='.vector_cache/wiki.en.vec')

'''
embeddingsはidでくるものをvectorにする。
'''

class LSTMClassifier(nn.Module):
    def __init__(self, text_id, hidden_dim, num_label):
        super(LSTMClassifier, self).__init__()
        self.gru_hidden_size = 64
        self.lstm_hidden_size = 300
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_id, freeze=True
        )
        self.embedding_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(self.lstm_hidden_size, hidden_dim, batch_first=True, bidirectional=True)

        self.gru = nn.GRU(hidden_dim * 2, self.gru_hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.gru_hidden_size * 6, 20)
        self.cls = nn.Linear(hidden_dim, num_label)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.gru_hidden_size * 6, num_label)
        # self.softmax = nn.LogSoftmax()

    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x):
        x_vec = self.embeddings(x)
        x_vec = self.apply_spatial_dropout(x_vec)

        h_lstm, lstm_out = self.lstm(x_vec)
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)

        # conc = self.linear(conc)

        # conc = self.relu(conc)
        conc = self.dropout(conc)

        out = self.out(conc)
        return out


def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')

class EMA:

    def __init__(self, model, mu, level='batch', n=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n
                
    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)


class ParamScheduler:

    def __init__(self, optimizer, scale_fn, step_size):
        if not isinstance(optimizer, Optimizer):
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


def eval_model(model, data_loader, is_train=False):
    all_labels = []
    all_preds = []
    for batch in data_loader:
        inputs = batch.text[0].to(device)
        if is_train:
            labels = batch.label.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            if is_train:
                all_labels += labels.tolist()
            all_preds += pred.tolist()

    return all_labels, all_preds


def train_model(model, dl_dict, criterion, optimizer, num_epochs):
    model.to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_n = int(len(dl_dict['train'].dataset) / (5 * BS))
    ema = EMA(model, 0.9, n=ema_n)
    scale_fn = combine_scale_functions(
        [partial(scale_cos, 1e-4, 5e-3), partial(scale_cos, 5e-3, 1e-2)], [0.2, 0.8])
    scheduler = ParamScheduler(optimizer, scale_fn, num_epochs * len(dl_dict["train"]))
    all_test_preds = list()
    all_oof_preds = list()
    all_oof_preds_ema = list()
    for epoch in range(num_epochs):
        all_loss = 0
        all_labels = []
        all_preds = []
        for batch in dl_dict['train']:
            inputs, inputs_length = batch.text[0].to(device), batch.text[1].to(device)  # 文章
            labels = batch.label.to(device)
            optimizer.zero_grad()

            ema.on_batch_end(model)
            
            scheduler.batch_step()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                all_loss += loss.item()
        print("train | epoch", epoch + 1, " | ", "loss", all_loss / len(dl_dict["train"]))
        all_labels, all_preds = eval_model(model, dl_dict["val"], is_train=True)
        all_oof_preds.append(all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        print("val | epoch", epoch + 1, " | ", "f1", train_f1)

        all_test_preds.append(eval_model(model, dl_dict["test"], is_train=False)[1])
    ema.set_weights(ema_model)
    ema_model.lstm.flatten_parameters()
    ema_model.gru.flatten_parameters()
    all_labels_ema, all_preds_ema = eval_model(ema_model, dl_dict["val"], is_train=True)
    all_oof_preds_ema.append(all_preds)
    ema_f1 = f1_score(all_labels_ema, all_preds_ema, average="macro")
    print('ema f1', ema_f1)
    checkpoint_weights = np.array([3 ** epoch for epoch in range(num_epochs)])
    checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()
    # eva/
    test_y = np.average(all_test_preds, weights=checkpoint_weights, axis=0).astype(float)
    oof = np.average(all_oof_preds, weights=checkpoint_weights, axis=0).astype(float)
    # test_y = np.round(test_y).astype(int)

    # test_y = np.mean([])

    return model, test_y, oof


train_ds, test_ds = torchtext.data.TabularDataset.splits(
    path=BASE_PATH, train='train_x.csv',
    test='test_x.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)
test_ds = torchtext.data.TabularDataset(
    path=BASE_PATH + "test_x.csv",
    format="csv",
    fields=[("text", TEXT)],
)
TEXT.build_vocab(train_ds, vectors=fasttext, min_freq=3)  # buildしないといけないらしいよくわからない
TEXT.build_vocab(test_ds, vectors=fasttext, min_freq=3)
kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
y_pred = np.zeros(test.shape[0])
oof = np.zeros(train.shape[0])
for fold, (tdx, vdx) in enumerate(kf.split(train_ds.examples)):
    print(fold + 1)
    data_arr = np.array(train_ds.examples)

    train_dl = torchtext.data.Iterator(Dataset(data_arr[tdx], fields=[("text", TEXT), ("label", LABEL)]), batch_size=BS,
                                       train=True)
    val_dl = torchtext.data.Iterator(Dataset(data_arr[vdx], fields=[("text", TEXT), ("label", LABEL)]), batch_size=BS,
                                     train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=BS, train=False, sort=False)
    dl_dict = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    
    model = LSTMClassifier(TEXT.vocab.vectors, 128, NUM_CLASS)
    # 損失関数
    weight = len(train) / train["label"].value_counts().sort_index().values
    weights = torch.tensor(weight.tolist()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.CrossEntropyLoss()
    # オプティマイザー
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model, test_y, oof_ = train_model(model, dl_dict, criterion, optimizer, NUM_EPOCHS)

    y_pred += test_y / N_FOLDS
    oof[vdx] = np.round(oof_.astype(float).astype(float)).astype(int)

y_pred = np.round(y_pred.astype(float)).astype(int)

language = "default"
columns = ["pred"]

lang_columns = [language + "_" + x for x in columns]

test_pred = pd.DataFrame(y_pred)
oof_pred = pd.DataFrame(oof)

test_pred.columns = lang_columns
oof_pred.columns = lang_columns
# test_pred.to_csv(f"../data/languages/test_{language}_lstm.csv")
# oof_pred.to_csv(f"../data/languages/train_{language}_lstm.csv")
# def make_submit_file(pred):
#     test_id = pd.read_csv(BASE_PATH + "test.csv")["id"]
#     submit = pd.DataFrame({'index': test_id, 'pred': pred + 1})
#     # aug = "using_aug" if augmentation else "non_aug"
#     submit.to_csv(f"../outputs/lstm_v1.csv", index=False, header=False)
#
# make_submit_file(y_pred)


print("DONE")
