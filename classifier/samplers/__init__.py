from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from classifier.dataset import DatasetEntry


class Sampler(ABC):
    name: str = "undefined"

    @abstractmethod
    def configure(self, config: dict):
        pass

    @abstractmethod
    def sample(
        self, train: list[DatasetEntry], test: list[DatasetEntry]
    ) -> list[DatasetEntry]:
        pass
