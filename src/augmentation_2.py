import pandas as pd
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
from pandarallel import pandarallel

import swifter
import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
df = pd.read_csv("../data/train.csv").iloc[:, 1:]

aug = naw.SynonymAug(
    # model_path="/",
    aug_src="wordnet"
)


pandarallel.initialize(progress_bar=True)

df_copy = df.copy()
df_copy["description"] = df_copy["description"].swifter.apply(aug.augment)

df_copy.to_csv("../data/train_augmented_2.csv", index=False)
