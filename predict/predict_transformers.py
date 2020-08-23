import pandas as pd
from train import bert_model

LANGUAGES = ["ja", "de", "fr", "default"]
MODELS = [["bert", "bert-base-uncased"],
          ["roberta", "roberta-base"],
          ["xlnet", "xlnet-base-cased"]]
DATA_PATH = "../for_train_data/"
SEED = 2020
N_FOLDS = 4

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
    "do_lower_case": True,
    "manual_seed": SEED,
    "verbose": False,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
}

bert_model.seed_everything(SEED)

columns = ["pred", "1_ce", "2_ce", "3_ce", "4_ce"]
for MODEL_TYPE, MODEL_NAME in MODELS:
    for language in LANGUAGES:
        for test_language in LANGUAGES:
            print(language, test_language)
            if test_language == "default":
                test = pd.read_csv(f"{DATA_PATH}test.csv").drop(['id'], axis=1)
            else:
                test = pd.DataFrame(pd.read_csv(f"{DATA_PATH}test_{test_language}_en.csv")["transrated"])
                test.columns = ["description"]

            params["output_dir"] = f"../models/{MODEL_NAME}_{language}"
            train = pd.read_csv(f"{DATA_PATH}train_{language}_base.csv")

            test_pred, f1_score, oof_pred = bert_model.model(train, test, params, N_FOLDS, MODEL_NAME, MODEL_TYPE,
                                                             LB_HACK,
                                                             prediction=True)
            print(language, test_language, "model predicting have done")

            lang_columns = [language + "_" + x for x in columns]

            test_pred.columns = lang_columns
            oof_pred.columns = lang_columns

            test_pred.to_csv(f"{DATA_PATH}languages/test_{language}_{test_language}_{MODEL_NAME}.csv")
            oof_pred.to_csv(f"{DATA_PATH}languages/train_{language}_{test_language}_{MODEL_NAME}.csv")
