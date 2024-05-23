from __future__ import annotations

import copy
import random
# from drfp import DrfpEncoder
import numpy as np

from rdkit.Chem import rdChemReactions
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from . import Sampler
from classifier.dataset import DatasetEntry


class TanimotoSampler(Sampler):
    name = "tanimoto"
    __predict = None
    __classes = None
    __seed = None
    __max_length = 5
    __request = None

    def configure(self, config: dict):
        self.__predict = config.get("class", self.__predict)
        self.__classes = config.get("classes", self.__classes)
        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__request = config.get("request", self.__request)
        self.__seed = config.get("seed", self.__seed)

    def sample(
            self,
            train: list[DatasetEntry],
            test: list[DatasetEntry],
    ) -> list[DatasetEntry]:

        train = copy.deepcopy(train)
        # random.seed(self.__seed)
        # random.shuffle(train)

        items = []

        request_rxn = rdChemReactions.ReactionFromSmarts(self.__request)
        request_fp = rdChemReactions.CreateStructuralFingerprintForReaction(request_rxn)

        similarities = []
        for i, entry in enumerate(train):
            if entry.features[0].lstrip('reaction: ') == self.__request:
                continue
            rxn = rdChemReactions.ReactionFromSmarts(entry.features[0].lstrip('reaction: '))
            example_fp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(request_fp, example_fp)
            similarities.append((i, tanimoto_similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        for sim in similarities:
            items.append(train[sim[0]])
            if len(items) >= self.__max_length:
                break

        return items
