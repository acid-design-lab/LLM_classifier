from __future__ import annotations

import requests.exceptions
from dotenv import load_dotenv

from enviroment.basics import create_result_directory
from enviroment.basics import parse
from estimate.metrics import calculate_classification_metrics
from gpt.gpt_funcs import create_request_template
from gpt.gpt_funcs import request_completion
from preparation.extracting import string_to_continuous
from preparation.load_dataset import load_dataset
from samplers import Sampler


load_dotenv()
# argv[1]
with open(r"./config", encoding="utf-8") as file:
    data_file = file.read()
    config = parse(data_file)

(
    sample_features_train,
    sample_features_test,
    sample_targets_train,
    sample_targets_test,
) = load_dataset(
    link=config["dataset"],
    targets=config["classes"],
    pure_text=config["is_text"],
    config=config,
)

result = []
history = []
log = []
missed_requests = []
path = create_result_directory(config=config)

for smp in Sampler.__subclasses__():
    if smp.name == config.get("sampler"):
        sampler = smp()
        sampler.configure(config)

for request in range(len(sample_features_test)):

    sampler.configure({"class": sample_targets_test[request].split(", ")[0]})
    samples = sampler.sample(
        list(zip(sample_features_train, sample_targets_train)),
        list(zip(sample_features_test, sample_targets_test)),
    )
    template = create_request_template(
        samples, seed=config["seed"], name=f"{config['subject']}_doctor"
    )

    try:
        completion_response = request_completion(
            question=sample_features_test[request],
            name=f"{config['subject']}_doctor",
            request=template,
        )
        print(completion_response.choices[0].message.content)

    except requests.exceptions.ConnectionError as e:
        log.append(e)
        with open(path / "log.txt", "w") as file:
            file.write("".join(log))
        raise

    except Exception as e:
        # raise
        log.append(
            f"Request_{request} missed\n"
            f"Content: {sample_features_test[request]}\n"
            f"Reason: {e}\n\n"
        )
        missed_requests.append(request)
        continue

    else:
        result.append(completion_response.choices[0].message.content)
        history.append(
            f"request: {sample_features_test[request]}\n"
            f"response: {completion_response.choices[0].message.content}\n"
            f"true: {sample_targets_test[request]}"
        )

if missed_requests:
    count = 0
    for i in missed_requests:
        del sample_targets_test[i - count]
        count += 1
    with open(path / "log.txt", "w") as file:
        file.write("".join(log))

full_data = {
    "train": sample_targets_train,
    "test": sample_targets_test,
    "valid": sample_targets_train + sample_targets_test,
}

for title in full_data.keys():
    calculate_classification_metrics(
        y_true=string_to_continuous(
            income_data=sample_targets_test, classes=config["classes"]
        ),
        y_pred=string_to_continuous(income_data=result, classes=config["classes"]),
        config=config,
        path=path,
        title=title,
    )

with open(path / "history.txt", "w") as file:
    file.write("".join(history))
