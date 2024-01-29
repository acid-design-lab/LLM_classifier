from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def calculate_classification_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, path: str, config: dict, title: str
):

    """This func calculates standard classification metrics and save it to ./experiments/your_name"""

    class_metrics = []

    for class_label in y_true.columns:
        precision = precision_score(
            y_true[class_label], y_pred[class_label], zero_division=1
        )

        recall = recall_score(y_true[class_label], y_pred[class_label], zero_division=1)

        f1 = f1_score(y_true[class_label], y_pred[class_label], zero_division=1)

        class_metrics.append(
            {
                "Class": class_label,
                "Accuracy": accuracy_score(y_true[class_label], y_pred[class_label]),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=1)

    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)

    micro_precision = precision_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    micro_recall = recall_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    micro_f1 = f1_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    class_metrics.append(
        {
            "Class": "Macro-Average",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1": macro_f1,
        }
    )

    class_metrics.append(
        {
            "Class": "Micro-Average",
            "Accuracy": accuracy_score(
                y_true.values.flatten(), y_pred.values.flatten()
            ),
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1": micro_f1,
        }
    )

    df_metrics = pd.DataFrame(class_metrics)
    df_metrics.to_csv(
        path / f'metrics_{title}_{config["nnet"]}_{config["data_format"]}.csv',
        index=False,
        sep=";",
        header=True,
        encoding="utf-8",
    )
