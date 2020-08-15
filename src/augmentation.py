import pandas as pd
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

import swifter

df = pd.read_csv("../data/train.csv").iloc[:, 1:]

aug = naw.WordEmbsAug(
    model_type="word2vec",
    model_path="../data/aug/GoogleNews-vectors-negative300",
    action="insert"
)

df_copy = df.copy()
# df_copy["description"] = df_copy["description"].apply(lambda x: aug.augment(x))
df_copy["description"] = df_copy["description"].swifter.apply(aug.augment)

print(0)

df_copy.to_csv("../data/train_augmented.csv", index=False)