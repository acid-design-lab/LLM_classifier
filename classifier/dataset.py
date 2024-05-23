from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from io import TextIOWrapper
from pathlib import Path
from typing import BinaryIO, Optional, Literal
from typing import TypeGuard

import pandas as pd
from sklearn.model_selection import train_test_split

from classifier.configuration import Configuration


@dataclass
class DatasetEntry:
    input_text: str
    output_text: str
    split: Optional[Literal["train"] | Literal["test"]] = None

    @property
    def features(self) -> list[str]:
        return self.input_text.split("; ")

    @property
    def classes(self) -> list[str]:
        return self.output_text.split(", ")

    @property
    def tuple(self):
        return self.input_text, self.output_text


def only_strings(item: str | None) -> TypeGuard[str]:
    return isinstance(item, str)


class Dataset(list[DatasetEntry]):

    def __init__(self, *args, has_predefined_split=False, **kwargs):
        self.has_predefined_split = has_predefined_split
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_io(data: str | Path | TextIOWrapper):
        if isinstance(data, str):
            return StringIO(data)
        if isinstance(data, Path):
            return open(data)
        if isinstance(data, TextIOWrapper):
            return data
        raise TypeError("Invalid data passed.")

    @classmethod
    def from_csv(
        cls,
        into_io: str | Path | TextIOWrapper,
        config: Configuration,
        delimiter: str = ",",
    ) -> Dataset:

        data: pd.DataFrame = pd.read_csv(cls._get_io(into_io), delimiter=delimiter)
        return cls._process_entries(data, config)

    @classmethod
    def from_xlsx(
        cls,
        path: Path | BinaryIO,
        config: Configuration,
    ) -> Dataset:

        if not isinstance(path, Path):
            raise TypeError("Invalid data passed.")
        data: pd.DataFrame = pd.read_excel(path)
        return cls._process_entries(data, config)

    @classmethod
    def load_path(
        cls,
        path: Path,
        config: dict | Configuration,
    ) -> Dataset:

        if isinstance(config, dict) and not isinstance(config, Configuration):
            config = Configuration(existing_data=config)
        if not isinstance(path, Path):
            raise TypeError(f"Invalid path: {path}.")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}.")
        if path.suffix == ".csv":
            return cls.from_csv(path, config)
        elif path.suffix == ".xlsx":
            return cls.from_xlsx(path, config)
        else:
            raise ValueError(f"Invalid file format: {path.suffix}")

    @classmethod
    def _process_entries(
        cls,
        _data: pd.DataFrame,
        config: Configuration,
    ) -> Dataset:

        has_predefined_split = False
        if "split" in _data.columns:
            has_predefined_split = True

        classes_labels: list[str] = config.classes
        (texts, classes) = _data.drop(classes_labels + ["split"] if has_predefined_split else [], axis=1), \
            _data[classes_labels]
        splits = None
        if has_predefined_split:
            splits = _data["split"].tolist()
        labels: list[str] = list(_data.drop(classes_labels + ["split"] if has_predefined_split else [], axis=1).columns)

        data: list[tuple[list[str], list[str], Optional[list[str]]]] = list(
            zip(
                map(lambda x: list(x[1:]), texts.itertuples()),
                map(lambda x: list(x[1:]), classes.itertuples()),
                *((splits,) if has_predefined_split else ())
            )
        )
        items = list(
            map(
                lambda entry: DatasetEntry(
                    input_text=" ".join(entry[0])
                    if config.pure_text
                    else (
                        "; ".join(
                            map(lambda x: f"{x[0]}: {x[1]}", zip(labels, entry[0]))
                        )
                    ),
                    output_text=", ".join(
                        list(
                            filter(
                                only_strings,
                                map(
                                    lambda arg: arg[1] if arg[0] else None,
                                    zip(entry[1], classes_labels),
                                ),
                            )
                        )
                    ),
                    split=entry[2] if has_predefined_split else None
                ),
                data,
            )
        )
        return cls(items, has_predefined_split=has_predefined_split)

    def train_test_split(
        self,
        test_size,
    ) -> tuple[Dataset, Dataset]:
        if self.has_predefined_split:
            train = list(filter(lambda x: x.split == "train", self))
            test = list(filter(lambda x: x.split == "test", self))
        else:
            train, test = train_test_split(self, test_size=test_size)
        return Dataset(train), Dataset(test)

    def tuples(self):
        return list(map(lambda x: x.tuple, self))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Dataset(list.__getitem__(self, idx))
        else:
            return list.__getitem__(self, idx)
