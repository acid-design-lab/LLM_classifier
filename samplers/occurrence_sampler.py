import copy
import random

from . import Sampler


class OccurrenceSampler(Sampler):
    name = "occurrence"
    __predict = None
    __classes = None
    __seed = None
    __max_length = 5

    def configure(self,
                  config: dict):

        self.__predict = config.get("class", self.__predict)
        self.__classes = config.get("classes", self.__classes)
        self.__max_length = config.get("n_for_train", self.__max_length)

    def sample(self,
               train: list[(str, str)],
               test: list[(str, str)],
               ) -> list[(str, str)]:

        train = copy.deepcopy(train)
        items = []

        for i, entry in enumerate(train):
            if self.__predict in entry[1].split(", "):
                items.append(entry)
                train.pop(i)
                break

        items += list(random.choices(train, k=(self.__max_length - 1)))
        return items

