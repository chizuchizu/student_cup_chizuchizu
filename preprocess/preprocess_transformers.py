import pandas as pd
from preprocess.replace import preprocessing_text

DATA_PATH = "../for_train_data/"


def preprocess(data, augment):
    if augment:
        data = data.drop(columns=['id', "description"])
        data = data.rename(columns={"transrated": 'text', "jobflag": 'label'})
    else:
        data = data.drop(columns=["id"])
        data = data.rename(columns={"description": "text", "jobflag": "label"})
    # data = data.rename(columns={"description": "text", "jobflag": "label"})
    data["text"] = data["text"].apply(preprocessing_text)
    data['label'] -= 1
    # 順序が違うとだめらしいので
    data = data.reindex(columns=["text", "label"])

    return data


LANGUAGES = ["ja", "de", "fr", "default"]
for language in LANGUAGES:
    print(language)
    if language != "default":
        train = preprocess(pd.read_csv(f"{DATA_PATH}{DATA_PATH}train_{language}_en.csv"), augment=True)
    else:
        train = preprocess(pd.read_csv(f"{DATA_PATH}train.csv"), augment=False)

    train.to_csv(f"../for_train_data/train_{language}_base.csv", index=False)
