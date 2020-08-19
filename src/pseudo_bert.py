"""
https://www.kaggle.com/sakami/single-lstm-3rd-place

1. train全データ学習でtestを予測
2. テストデータでKfoldしリークしないようにtestを再予測

今の所trainもkfoldしている（stacking用）が、その必要は果たして必要なのか……　私にはわからない。
"""

import pandas as pd
from src import bert_model

LANGUAGES = ["ja", "de", "fr", "default"]

DATA_PATH = "../data/"
SEED = 2020
N_FOLDS = 4
MODEL_TYPE = "roberta"
MODEL_NAME = "roberta-base"
# MODEL_TYPE = "xlnet"
# MODEL_NAME = "xlnet-base-cased"  # xxlarge-v2
LB_HACK = False
params = {
    # "output_dir": "outputs/",
    "max_seq_length": 64,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "reprocess_input_data": True,
    # "do_lower_case": True,
    "manual_seed": SEED,
    "verbose": False,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
}

bert_model.seed_everything(SEED)

test = pd.read_csv(f"{DATA_PATH}test.csv").drop(['id'], axis=1)


def preprocess(data, augment=True):
    if augment:
        data = data.drop(columns=['id', "description"])
        data = data.rename(columns={"transrated": 'text', "jobflag": 'label'})
    else:
        data = data.drop(columns=["id"])
        data = data.rename(columns={"description": "text", "jobflag": "label"})
    # data = data.rename(columns={"description": "text", "jobflag": "label"})
    data['label'] -= 1
    # 順序が違うとだめらしいので
    data = data.reindex(columns=["text", "label"])

    return data


"""全データ学習"""

columns = ["pred", "1_ce", "2_ce", "3_ce", "4_ce"]
for MODEL_TYPE, MODEL_NAME in zip(["bert", "roberta", "xlnet"], ["bert-base-uncased", "roberta-base", "xlnet-base-cased"]):
    for language in LANGUAGES:
        print(language)
        if language != "default":
            train = preprocess(pd.read_csv(f"{DATA_PATH}{DATA_PATH}train_{language}_en.csv"))
        else:
            train = preprocess(pd.read_csv(f"{DATA_PATH}train.csv"), augment=False)
        """pseudo labeling"""

        print("all_data")
        test_pred, pseudo_idx = bert_model.all_train(train, test, params, MODEL_NAME, MODEL_TYPE, LB_HACK)
        # test_pred.to_csv(f"{DATA_PATH}languages/test_{language}_{MODEL_NAME}_{LB_HACK}.csv"
        test["jobflag"] = test_pred.copy()

        print("pseudo labeling")
        for_pseudo_test = test.copy().rename(columns={"description": "text", "jobflag": "label"}).loc[pseudo_idx]
        test_pred, f1_score, oof_pred = bert_model.cross_pseudo_labeling(train, for_pseudo_test, params, N_FOLDS,
                                                                         MODEL_NAME, MODEL_TYPE, LB_HACK)
        print(language, "model training have done")

        lang_columns = [language + "_" + x for x in columns]

        test_pred.columns = lang_columns
        oof_pred.columns = lang_columns

        test_pred.to_csv(f"{DATA_PATH}languages/test_{language}_{MODEL_NAME}_{LB_HACK}_pseudo.csv")
        oof_pred.to_csv(f"{DATA_PATH}languages/train_{language}_{MODEL_NAME}_{LB_HACK}_pseudo.csv")
