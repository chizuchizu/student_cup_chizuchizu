import pandas as pd
from src import bert_model

LANGUAGES = ["ja", "de", "fr", "default"]

DATA_PATH = "../data/"
SEED = 2020
N_FOLDS = 4
MODEL_TYPE = "bert"
MODEL_NAME = "bert-base-uncased"
params = {
    # "output_dir": "outputs/",
    "max_seq_length": 64,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "reprocess_input_data": True,
    "do_lower_case": True,
    "manual_seed": SEED,
    "verbose": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
}

bert_model.seed_everything(SEED)

# TODO: 言語ごとのテストデータを作る
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


columns = ["pred", "1_ce", "2_ce", "3_ce", "4_ce"]
for language in LANGUAGES:
    print(language)
    if language != "default":
        train = preprocess(pd.read_csv(f"{DATA_PATH}{DATA_PATH}train_{language}_en.csv"))
    else:
        train = preprocess(pd.read_csv(f"{DATA_PATH}train.csv"), augment=False)

    test_pred, f1_score, oof_pred = bert_model.model(train, test, params, N_FOLDS, MODEL_NAME, MODEL_TYPE)
    print(language, "model training have done")

    lang_columns = [language + "_" + x for x in columns]

    test_pred.columns = lang_columns
    oof_pred.columns = lang_columns

    test_pred.to_csv(f"{DATA_PATH}languages/test_{language}.csv")
    oof_pred.to_csv(f"{DATA_PATH}languages/train_{language}.csv")
