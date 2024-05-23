from __future__ import annotations

import copy
import random
from pprint import pprint

from . import Sampler
from classifier.dataset import DatasetEntry


class SimpleSampler(Sampler):
    name = "simple"
    __predict = None
    __classes = None
    __max_length = 5

    def configure(self, config: dict):

        self.__predict = config.get("class", self.__predict)
        self.__classes = config.get("classes", self.__classes)
        self.__max_length = config.get("n_for_train", self.__max_length)

    def sample(
        self, train: list[DatasetEntry], test: list[DatasetEntry]
    ) -> list[DatasetEntry]:

        train = copy.deepcopy(train)
        items = []

        # for i, entry in enumerate(train):
        #     if self.__predict in entry.classes:
        #         items.append(entry)
        #         train.pop(i)
        #         break
        #     if i + 1 == len(train):
        #         raise ValueError(
        #             f"Impossible to generate dataset: class {self.__predict} doesn't exist in train dataset"
        #         )
        n_high = self.__max_length // 2
        n_low = self.__max_length - n_high
        high = list(filter(lambda x: "high_yielding" in x.classes, train))
        low = list(filter(lambda x: "not_high_yielding" in x.classes, train))
        high = random.choices(high, k=n_high)
        low = random.choices(low, k=n_low)
        return [*low, *high]
