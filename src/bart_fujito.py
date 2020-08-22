import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from transformers import BartTokenizer, BartModel,BertModel,BertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user', type=str,help='user name', required=True)
config = parser.parse_args()
user = config.user
if user == 'chizuchizu':
    BASE_PATH ='../data/'
else:
    BASE_PATH = './data/'

train = pd.read_csv(BASE_PATH + 'train.csv')
print(train)