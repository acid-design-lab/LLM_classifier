from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_features(data: list, config: dict) -> (list, list, list, list):
    """This func results to the str type data of the columns of experimental conditions
    and target variables in the case when they are represented in a feature-style table."""

    features_train, features_test, targets_train, targets_test = train_test_split(
        data[0], data[1], test_size=config["test_size"], random_state=config["seed"]
    )

    for form in targets_train.columns:
        targets_train[form] = targets_train[form].apply(
            lambda x: form if x == 1 else None
        )
        targets_test[form] = targets_test[form].apply(
            lambda x: form if x == 1 else None
        )

    for feat in features_train.columns:
        features_train[feat] = features_train[feat].apply(lambda x: f"{feat}: {x}")
        features_test[feat] = features_test[feat].apply(lambda x: f"{feat}: {x}")

    sample_targets_train = [
        ", ".join([i for i in targets_train.values[j] if i])
        for j in range(len(targets_train.values))
    ]
    sample_targets_test = [
        ", ".join([i for i in targets_test.values[j] if i])
        for j in range(len(targets_test.values))
    ]
    sample_features_train = [
        "; ".join(features_train.values.tolist()[i])
        for i in range(len(features_train.values))
    ]
    sample_features_test = [
        "; ".join(features_test.values.tolist()[i])
        for i in range(len(features_test.values))
    ]

    return (
        sample_features_train,
        sample_features_test,
        sample_targets_train,
        sample_targets_test,
    )


def load_dataset(
    link: str,
    targets: list,
    config: dict,
    pure_text: bool = False,
) -> (list, list, list, list):
    """This func results to the str type data of the columns of experimental conditions
    and target variables in the case when they are represented in a human-speech-style table."""

    if ".csv" in link:
        data = pd.DataFrame(
            pd.read_csv(
                link,
                # index_col="idx"
            )
        )
    elif ".xlsx" in link:
        data = pd.DataFrame(pd.read_excel(link))
    else:
        raise FileNotFoundError("File is not exists.")

    classes = pd.DataFrame(data[targets])
    data = data.drop(targets, axis=1)

    if not pure_text:
        return extract_features([data, classes], config)

    # data = data.drop(["Unnamed: 0"], axis=1) # костыль

    features_train, features_test, classes_train, classes_test = train_test_split(
        data, classes, test_size=config["test_size"], random_state=config["seed"]
    )

    sample_features_train = [
        "".join([str(i) for i in data.values[j] if i])
        for j in range(features_train.shape[0])
    ]
    sample_features_test = [
        "".join([str(i) for i in data.values[j] if i])
        for j in range(features_test.shape[0])
    ]

    for feat in classes.columns:
        classes_train[feat] = classes_train[feat].apply(
            lambda x: feat if x == 1 else None
        )
        classes_test[feat] = classes_test[feat].apply(
            lambda x: feat if x == 1 else None
        )

    sample_targets_train = [
        ", ".join([str(i) for i in classes_train.values.tolist()[j] if i])
        for j in range(classes_train.shape[0])
    ]
    sample_targets_test = [
        ", ".join([str(i) for i in classes_test.values.tolist()[j] if i])
        for j in range(classes_test.shape[0])
    ]
    print(1)
    return (
        sample_features_train,
        sample_features_test,
        sample_targets_train,
        sample_targets_test,
    )
