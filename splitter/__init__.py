from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class TrainTestSplitter(ABC):
    @abstractmethod
    def configure(self, config):
        pass

    @abstractmethod
    def apply(self):
        pass
