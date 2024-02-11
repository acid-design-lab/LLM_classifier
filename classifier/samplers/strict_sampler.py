from __future__ import annotations

import copy
import random

from . import Sampler
from classifier.dataset import DatasetEntry


class StrictSampler(Sampler):
    name = "strict"
    __max_length = 5
    __predict = None
    __seed = None

    def configure(self, config: dict):

        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__predict = config.get("class", self.__predict)
        self.__seed = config.get("seed", self.__seed)

    def sample(
        self,
        train: list[DatasetEntry],
        test: list[DatasetEntry],
    ) -> list[DatasetEntry]:

        items = []
        train = copy.deepcopy(train)
        random.seed(self.__seed)
        random.shuffle(train)

        for entry in train:
            if self.__predict in entry.classes:
                items.append(entry)
            if len(items) >= self.__max_length:
                break

        return items
