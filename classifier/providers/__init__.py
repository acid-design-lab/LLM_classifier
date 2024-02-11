from __future__ import annotations

import abc
from dataclasses import dataclass

from classifier.dataset import DatasetEntry


@dataclass
class CompletionRequest:
    samples: list[DatasetEntry]
    question: str
    engine: str


@dataclass
class CompletionResponse:
    text: str | None
    cost: float | None

    @property
    def classes(self):
        return self.text.split(", ")


class CompletionProvider(abc.ABC):
    @abc.abstractmethod
    def get_completion(
        self, request: CompletionRequest, dry_run: bool = False
    ) -> CompletionResponse:
        pass

    def configure(self, configuration: dict) -> None:
        pass
