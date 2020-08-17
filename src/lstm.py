import string
import random
import pandas as pd
from sklearn.model_selection import GroupKFold

import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors

#confing
SEED = 2020
random.seed(SEED)
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

train.to_csv(BASE_PATH + 'train_x.csv',index = False, header=False)
test.to_csv(BASE_PATH + 'test_x.csv',index = False, header=False)

def preprocessing_text(text):
    for p in string.punctuation:
        text = text.replace(p,'')
    text = text.strip().split()
    return text

TEXT = torchtext.data.Field(sequential=True, 
                            tokenize=preprocessing_text,
                            use_vocab = True,
                            fix_length= 64,
                            lower = True)

LABEL = torchtext.data.Field(sequential=False,
                             use_vocab = False)

train_ds, test_ds = torchtext.data.TabularDataset.splits(
    path = BASE_PATH, train = 'train_x.csv',
    test = 'test_x.csv', format = 'csv',
    fields = [('text',TEXT), ('label', LABEL)]
)

# traindataをvalとtrainに分割(あとでcvをじっそうしたい)
train_ds, val_ds = train_ds.split(split_ratio=0.8, random_state=random.seed(SEED))
# print(vars(train_ds[0]))

#一回
first = False
if first:
    fasttext = torchtext.vocab.FastText(language="en")#分かち書きをvecotr化するここをfnとかにしたらフランス語に対応できるかも？    
else:
    fasttext = Vectors(name = '.vector_cache/wiki.en.vec')

TEXT.build_vocab(train_ds, vectors=fasttext, min_freq=10)#buildしないといけないらしいよくわからない

#dataloaderの作成(cv実装したい)
train_dl = torchtext.data.Iterator(train_ds, batch_size = 32, train = True)
val_dl = torchtext.data.Iterator(val_ds, batch_size = 32, train = False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size = 32, train = False, sort = False)
dl_dict = {'train':train_dl,'val':val_dl,'test':test_dl}

'''
embeddingsはidでくるものをvectorにする。
'''

class LSTMClassifier(nn.Module):
    def __init__(self, text_id,hidden_dim,num_label):
        super(LSTMClassifier, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings = text_id, freeze=True
        )
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True)
        self.cls = nn.Linear(hidden_dim, num_label)
        # self.softmax = nn.LogSoftmax()
    def forward(self, x):
        x_vec = self.embeddings(x)
        _, lstm_out = self.lstm(x_vec)
        out = self.cls(lstm_out[0].squeeze())
        # pred = self.softmax(out.squeeze())
        return out

    

model = LSTMClassifier(TEXT.vocab.vectors,128,NUM_CLASS)
#損失関数
criterion = nn.CrossEntropyLoss()
#オプティマイザー
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, dl_dict, criterion, optimizer,num_epochs):
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            for batch in (dl_dict[phase]):
                inputs = batch.text.to(device)  # 文章
                labels = batch.label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) 
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
    return net
# batch = next(iter(val_dl))
# x = model(batch.text)
# print(x)
