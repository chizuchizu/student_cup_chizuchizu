import pandas as pd
import torch
import re
# from transformers.modeling_bert import BertModel
from transformers import BertTokenizer, BertForPreTraining, BertModel, BertForSequenceClassification
from sklearn.metrics import precision_score
import torch
from torch import nn
import json
import torchtext
import string
import random
import sys
import re
import numpy as np
import gc
from tqdm import tqdm
# import xgboost as xgb

from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from functools import partial

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
# from matplotlib_venn import venn2
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

weight = len(df_train) / df_train["jobflag"].value_counts().sort_index().values
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
del df_train['id']
df_train.drop_duplicates(subset='description', keep='last', inplace=True)
df_train = df_train.reset_index(drop=True)
df_train['id'] = np.arange(2905)
df_train["jobflag"] -= 1

tokenizer_bert = BertTokenizer(vocab_file='../data/bert-base-uncased-vocab.txt')


def preprocessing(text):
    text = text.lower()
    return text


def token_same_len(token):
    token.insert(0, '[CLS]')
    if len(token) < 128:
        while len(token) != 128:
            token.insert(128, '[PAD]')
    token.insert(127, '[SEP]')
    return token


def token_and_prepro(text, tokenizer=tokenizer_bert):
    text = preprocessing(text)
    token = tokenizer.tokenize(text)
    token = token_same_len(token)
    ids = tokenizer.convert_tokens_to_ids(token[:128])
    return ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, token_same_len):
        self.df = df
        self.tokenizer = tokenizer
        self.token_same_len = token_same_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, input_ids):
        text = self.df['description'][input_ids]
        label = self.df['jobflag'][input_ids]
        id = self.df['id'][input_ids]

        token = self.tokenizer(text)

        tensor_token = torch.tensor(token)
        tensor_label = torch.tensor(label)

        return tensor_token, tensor_label, id


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, token_same_len):
        self.df = df
        self.tokenizer = tokenizer
        self.token_same_len = token_same_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, input_ids):
        text = self.df['description'][input_ids]
        id = self.df['id'][input_ids]

        token = self.tokenizer(text)

        tensor_token = torch.tensor(token)

        return tensor_token, id


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label, id in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


weight = len(df_train) / df_train["jobflag"].value_counts().sort_index().values
model = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=768, out_features=4)
        #         self.dropout = nn.Dropout(0.3)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.classifier.bias, 0)

    def forward(self, inputs):
        output = self.bert(inputs)
        vec_0 = output[0]
        vec_0 = vec_0[:, 0, :]
        vec_0 = vec_0.view(-1, 768)

        vec_0 = self.dropout(vec_0)
        result = self.classifier(vec_0)

        return result


bert_model = BERT()
bert_model.train()

# 1.勾配計算Falseにする（ALl）
for param in bert_model.parameters():
    param.requires_grad = True
# 2. BertLayer[12]そう目
for param in bert_model.bert.encoder.layer[-1].parameters():
    param.requires_grad = True
# 3. label
for param in bert_model.classifier.parameters():
    param.requires_grad = True


def metric_f1(preds, data):
    """
    F1 evaluation function for lgbm model.
    """
    y_true = data.get_label()
    preds = preds.reshape(4, len(preds) // 4)
    y_pred = np.argmax(preds, axis=0)
    score = f1_score(y_true, y_pred, average="macro")
    return "metric_f1", score, True


def train_model(net, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    # batch_size = dataloaders_dict["train"].batch_size
    batch_size = BS
    a = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            f1_batch = 0
            epoch_corrects = 0
            iteration = 1
            c = 0

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                # GPUが使えるならGPUにデータを送る
                inputs = batch[0].to(device)  # 文章
                labels = batch[1].to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BERTに入力
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if iteration % 10 == 0:  # 10iterに1度、lossを表示
                            a += 1
                            # print(a)
                            scheduler.batch_step()
                            acc = (torch.sum(preds == labels.data)
                                   ).double() / batch_size
                            f1 = precision_score(preds.cpu().numpy(),
                                                 labels.data.cpu().numpy(),
                                                 average='macro')
                            print(' All / batch　{} || Loss: {:.4f} | ACC：{}　| F1 :{}'.format(
                                iteration, loss.item(), acc, f1))

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)
                    f1_batch += f1_score(preds.to('cpu').detach().clone().numpy(),
                                         labels.data.to('cpu').detach().clone().numpy(),
                                         average='macro')
                    # print(f1_batch)
                    # if c == 26:
                    #     sys.exit()
                    # c += 1
                    # print(c)
                    # f1_batch += precision_score(preds.cpu().numpy(),
                    #                             labels.data.cpu().numpy(),
                    #                             average='macro')
                    # print(f1_batch)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
            f1s = f1_batch / len(dataloaders_dict[phase])
            epoch_f1 = f1_batch / len(dataloaders_dict[phase])

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}  F1: {}'.format(epoch + 1, num_epochs,
                                                                                   phase, epoch_loss, epoch_acc,
                                                                                   epoch_f1))

    return net


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


num_epochs = 6
BS = 32

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': bert_model.bert.encoder.layer[-1].parameters(), 'lr': 1e-4},
    {'params': bert_model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=0.02)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

scale_fn = combine_scale_functions(
    [partial(scale_cos, 1e-4, 5e-3), partial(scale_cos, 5e-3, 1e-3)], [0.2, 0.8])
scheduler = ParamScheduler(optimizer, scale_fn, num_epochs * len(df_train))

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4, 6], gamma=0.5)
# 損失関数の設定
# weights = torch.tensor(weight).cuda()
# print(weights)
# weights = torch.tensor([2.192926045016077,4.011764705882353,1.0,2.3557858376511227]).cuda()
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to("cuda:0"))
# criterion = nn.MultiMarginLoss()

dataset = Dataset(df=df_train, tokenizer=token_and_prepro, token_same_len=token_same_len)
test_dataset = TestDataset(df=df_test, tokenizer=token_and_prepro, token_same_len=token_same_len)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
train_sampler = BalancedBatchSampler(train_dataset, 4, 4)
# train
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)
# train_dataloader = torch.utils.data.DataLoader(train_dataset,  batch_size= 64, shuffle=True)
# val
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BS, shuffle=False)
# test
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=False)
# loader_dict
load_dict = {"train": train_dataloader, 'val': val_dataloader, 'test': dl_test}

num_epochs = 6
net_trained = train_model(bert_model,
                          load_dict,
                          criterion,
                          optimizer,
                          scheduler,
                          num_epochs=num_epochs)


def make_df(phase):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds_list = []
    label_list = []
    outputs1_list = []
    outputs2_list = []
    outputs3_list = []
    outputs4_list = []
    id_list = []
    df_error = pd.DataFrame()
    df_errors = pd.DataFrame()

    if phase == 'test':
        bert_model.eval()
        bert_model.to(device)
        for batch in tqdm(load_dict[phase]):
            inputs = batch[0].to(device)
            id_list.extend(batch[1].to('cpu').detach().clone().numpy())

            with torch.set_grad_enabled(False):
                outputs_ = net_trained(inputs)
                # loslogits = outputs
                _, preds = torch.max(outputs_, 1)  # ラベルを予測
                preds_list.extend(preds.to('cpu').detach().clone().numpy())
                outputs = torch.t(outputs_).to('cpu').detach().clone().numpy()
                outputs1_list.extend(outputs[0])
                outputs2_list.extend(outputs[1])
                outputs3_list.extend(outputs[2])
                outputs4_list.extend(outputs[3])
                # label_list.extend(labels.to('cpu').detach().clone().numpy())
                df_error = pd.DataFrame({'prob0': outputs1_list,
                                         'prob1': outputs2_list,
                                         'prob2': outputs3_list,
                                         'prob3': outputs4_list,
                                         'id': id_list,
                                         'pred': preds_list})
        df_errors = pd.merge(df_error, df_test, on='id')
        body_len = []
        # print('a')
        for i in df_errors['description']:
            body_len.append(len(i))
        df_errors['len'] = body_len
        train_x = df_errors[['id', 'prob0', 'prob1', 'prob2', 'prob3', 'pred', 'description']]
        return train_x

    else:
        bert_model.eval()
        bert_model.to(device)
        for batch in tqdm(load_dict[phase]):
            inputs = batch[0].to(device)
            id_list.extend(batch[2].to('cpu').detach().clone().numpy())
            labels = batch[1].to(device)

            with torch.set_grad_enabled(False):
                outputs_ = net_trained(inputs)
                # loslogits = outputs
                _, preds = torch.max(outputs_, 1)  # ラベルを予測
                preds_list.extend(preds.to('cpu').detach().clone().numpy())
                outputs = torch.t(outputs_).to('cpu').detach().clone().numpy()
                outputs1_list.extend(outputs[0])
                outputs2_list.extend(outputs[1])
                outputs3_list.extend(outputs[2])
                outputs4_list.extend(outputs[3])
                label_list.extend(labels.to('cpu').detach().clone().numpy())

        df_error = pd.DataFrame({'prob0': outputs1_list,
                                 'prob1': outputs2_list,
                                 'prob2': outputs3_list,
                                 'prob3': outputs4_list,
                                 'id': id_list,
                                 'label': label_list,
                                 'pred': preds_list})
        df_errors = pd.merge(df_error, df_train, on='id')
        body_len = []
        # print('a')
        for i in df_errors['description']:
            body_len.append(len(i))
        df_errors['len'] = body_len
        train_x = df_errors[['id', 'prob0', 'prob1', 'prob2', 'prob3', 'pred']]
        train_y = df_errors['label']
        return train_x, train_y


val_x, val_y = make_df('val')
# train_x, train_y = make_df('train')
test_x = make_df('test')
print(f1_score(val_y, val_x['pred'], average='macro'))

print("Done")
