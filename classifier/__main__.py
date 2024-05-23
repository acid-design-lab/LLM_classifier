from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from pprint import pprint
from sys import argv

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import requests.exceptions
import tqdm
from dotenv import load_dotenv

from .configuration import Configuration
from .configuration import ConfigurationError
from .dataset import Dataset
from .logger import logger
from .providers import CompletionProvider
from .providers import CompletionRequest
from .providers.mistral_provider import MistralCompletionProvider
from .providers.openai_provider import OpenAICompletionProvider
from .providers.anthropic_provider import AnthropicCompletionProvider
from .providers.sber_provider import SberCompletionProvider
from .providers.yandex_provider import YandexGPTCompletionProvider
from .samplers.simple_sampler import SimpleSampler
from .utils import create_result_directory
from classifier.samplers.occurrence_sampler import OccurrenceSampler
from classifier.samplers.strict_sampler import StrictSampler
from classifier.samplers.tanimoto_sampler import TanimotoSampler
from classifier.samplers.equal_sampler import EqualSampler

experimental_features = {
    "smiles2image": False,
    "s3storage": False,
}
load_dotenv()

if "--experimental-s2i" in argv:
    experimental_features["smiles2image"] = True
if "--experimental-s3s" in argv:
    experimental_features["s3storage"] = True

config_path = argv[1]
config = Configuration.load(Path(config_path))
random.seed(int(config.seed))
dataset = Dataset.load_path(config.dataset, config)
splits: tuple[Dataset, Dataset] = dataset.train_test_split(test_size=config.test_size)
train_dataset, test_dataset = splits
test_dataset = test_dataset[:10]  # To delete when you gonna use it!!!!!!

results: list[dict] = []
path = create_result_directory(config=config)
total_cost: float = 0

sampler = None
for smp in [
    StrictSampler,
    OccurrenceSampler,
    TanimotoSampler,
    EqualSampler,
    SimpleSampler
]:
    if smp.name == config.get("sampler"):
        sampler = smp()
        sampler.configure(config)

if sampler is None:
    logger.error("No sampler specified.")
    raise ConfigurationError("sampler")
err = None


if experimental_features["s3storage"]:
    from .s3storage import S3Storage
    storage = S3Storage()
if experimental_features["smiles2image"]:
    from .smiles2image import Smiles2ImageConverter
    converter = Smiles2ImageConverter(save_path=path)
for dry_run in [True,
                False]:
    for item in tqdm.tqdm(test_dataset):
        sampler.configure({"class": item.classes[0],
                           "request": item.input_text})
        samples = sampler.sample(train_dataset,  # + test_dataset,
                                 test_dataset)
        provider: CompletionProvider | None = None
        if config.provider == "openai":
            provider = OpenAICompletionProvider()
        elif config.provider == "mistral":
            provider = MistralCompletionProvider()
        elif config.provider == "anthropic":
            provider = AnthropicCompletionProvider()
        elif config.provider == "yandex":
            provider = YandexGPTCompletionProvider()
        elif config.provider == "sber":
            provider = SberCompletionProvider()

        if provider is None:
            logger.error("No provider specified.")
            raise ConfigurationError("provider")

        provider.configure(config)
        if experimental_features["s3storage"]:
            provider.configure({"store_fn": storage.store})
        if experimental_features["smiles2image"]:
            provider.configure({"vision": True, "convert_fn": converter.convert})
        try:
            if not dry_run:
                # time.sleep(1)
                logger.debug(f'Trying to get completion for "{item.input_text}"')
            completion = provider.get_completion(
                CompletionRequest(
                    samples=samples, question=item.input_text, engine=config.engine
                ),
                dry_run=dry_run,
            )
            if not dry_run:
                logger.debug(
                    f"Successfully retrieved completion. Classes: {completion.classes}"
                )
            logger.debug(f"Approximate cost of request is {completion.cost:.02f} $ / € / ₽")
            if not dry_run:
                results.append(
                    dict(
                        input=item.input_text,
                        target_classes=item.classes,
                        predicted_classes=completion.classes,
                    )
                )
            if completion.cost:
                total_cost += completion.cost
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during completion generation: {e}")
            logger.error(f"Item content: {item.input_text}")
            continue
        except Exception as e:
            err = e
            break

    if dry_run:
        print(f"Total cost will be approximately {total_cost:.02f}$")

        if input("Do you want to continue? (y/n) ").lower() != "y":
            exit(0)
        total_cost = 0

if dataset.has_predefined_split:
    if config.get("enable_metrics") and len(results) != 0:
        y_true = pd.DataFrame(results)["target_classes"].apply(
            lambda x: 1 if "".join(x).lower().replace("-", "_") == "high_yielding" else 0)

        y_pred = pd.DataFrame(results)["predicted_classes"].apply(
            lambda x: 1 if "".join(x).lower().replace("-", "_") == "high_yielding" else 0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        metrics = pd.DataFrame([[accuracy, f1, recall, precision]], columns=["accuracy", "f1", "recall", "precision"])
        metrics.to_csv(path / "metrics.csv", sep="\t", index=False)

with open(path / "results.json", "w") as file:
    json.dump(
        list(results),
        file,
    )
shutil.copy(config_path, path / "config")
logger.info(f"Total cost: {total_cost:.02f}$")
if err:
    raise err
