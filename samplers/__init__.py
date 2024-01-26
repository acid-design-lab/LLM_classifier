from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class Sampler(ABC):
    name = None

    @abstractmethod
    def configure(self, config: dict):
        pass

    @abstractmethod
    def sample(self, train: list[str], test: list[str]) -> list:
        pass
