import argparse
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torchtext
from transformers import BartTokenizer, BartModel,BertModel,BertTokenizer

model_base = BartModel.from_pretrained('facebook/bart-base')
model_large = BartModel.from_pretrained('facebook/bart-large')
#config
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user', type=str,help='user name', required=True)
config = parser.parse_args()
user = config.user
if user == 'chizuchizu':
    BASE_PATH ='../data/'
else:
    BASE_PATH = './data/'
TEXT_COL = "description"
TARGET = "jobflag"
NUM_CLASS = 4
N_FOLDS = 4
BS = 256
NUM_EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#load data 
train = pd.read_csv(BASE_PATH + "train.csv").drop(['id'], axis=1)
train = train.rename(columns={TEXT_COL: 'text', TARGET: 'label'})
train['label'] -= 1

test = pd.read_csv(BASE_PATH + "test.csv").drop(['id'], axis=1)
test = test.rename(columns={TEXT_COL: 'text', TARGET: 'label'})


#dataset
#datalaoder
#make bart
class BART(nn.Module):
    def __init__(self):
        super(BART, self).__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-large')

#make eval
def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')
#training
#pred
