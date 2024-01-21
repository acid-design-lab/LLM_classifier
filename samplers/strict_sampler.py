import copy
import random
from samplers import Sampler


class StrictSampler(Sampler):
    name = "strict"
    __max_length = 5
    __predict = None
    __seed = None

    def configure(self,
                  config: dict):

        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__predict = config.get("class", self.__predict)
        self.__seed = config.get("seed", self.__seed)

    def sample(self,
               train: list[(str, str)],
               test: list[(str, str)]
               ) -> list[(str, str)]:

        items = []
        train = copy.deepcopy(train)
        random.seed(self.__seed)
        random.shuffle(train)

        random.seed(self.__seed)
        for entry in train:
            if self.__predict in entry[1].split(", "):
                items.append(entry)
            if len(items) >= self.__max_length:
                break

        return items
