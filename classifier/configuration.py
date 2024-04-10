from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import no_type_check
from typing import Self
from typing import Union


ValueType = Union[int, float, str, Path]


class Configuration(dict[str, str | int | bool | Path]):
    def __getattr__(self, item: str) -> Any:
        try:
            return super(dict, Configuration).__getattribute__(self, item)
        except AttributeError:
            return self[item]

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            raise ConfigurationError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    @no_type_check
    def load(cls, path: Path) -> Self:
        if not isinstance(path, Path):
            raise ValueError(f"Invalid argument: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        content = open(path).read()

        lines = map(lambda x: x.split("=", 1), content.splitlines())
        data: Self = cls()
        for k, v in lines:
            data[k] = v.strip()

        data["dataset"] = Path(data["dataset"]).absolute()
        data["classes"] = data.get("classes", "").split(",")
        data["pure_text"] = data.get("data_format", "text") != "table"
        data["n_for_train"] = int(data.get("n_for_train"))
        data["test_size"] = float(data.get("test_size"))
        data["seed"] = int(data.get("seed"))

        return data

    def __init__(self, existing_data=None):
        super().__init__()
        if existing_data:
            self.update(existing_data)


class ConfigurationError(Exception):
    def __init__(self, field: str) -> None:
        self.field: str = field

    def __str__(self) -> str:
        return f"Configuration error: {self.field} is not present or set not properly."
