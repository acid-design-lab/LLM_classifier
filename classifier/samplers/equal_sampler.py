from __future__ import annotations

import copy
import random

from . import Sampler
from classifier.dataset import DatasetEntry


class EqualSampler(Sampler):
    name = "equal"
    __predict = None
    __classes = None
    __max_length = 5
    __seed = None
    __chosen_entries = None

    def configure(self, config: dict):

        self.__predict = config.get("class", self.__predict)
        self.__classes = config.get("classes", self.__classes)
        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__seed = config.get("seed", self.__seed)

    def sample(
        self, train: list[DatasetEntry], test: list[DatasetEntry]
    ) -> list[DatasetEntry]:

        if self.__chosen_entries is not None:
            return self.__chosen_entries
        train = copy.deepcopy(train)
        items = []

        for i, entry in enumerate(train):
            if not items:
                items.append(entry)
                train.pop(i)
            else:
                if items[0].classes != entry.classes:
                    items.append(entry)
                    train.pop(i)
                    break
            # if i + 1 == len(train):
            #     raise ValueError(
            #         f"Impossible to generate dataset: class {self.__predict} doesn't exist in train dataset"
            #     )

        random.seed(self.__seed)
        items += list(random.choices(train, k=(self.__max_length - 2)))
        self.__chosen_entries = items
        return items
