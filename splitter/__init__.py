from abc import ABC, abstractmethod


class TrainTestSplitter(ABC):
    @abstractmethod
    def configure(self, config):
        pass

    @abstractmethod
    def apply(self):
        pass


