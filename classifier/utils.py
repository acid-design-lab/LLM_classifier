from __future__ import annotations

from pathlib import Path


def create_result_directory(config: dict) -> Path:

    experiments_directory = Path("./experiments")
    (experiments_directory / config["name"]).mkdir(parents=True, exist_ok=True)
    user_directory = experiments_directory / config["name"]
    net_directory = user_directory / config["provider"]
    dir_number = max(list(map(lambda x: int(x.stem), net_directory.glob("*"))) + [0])
    result_directory = net_directory / str(dir_number + 1)
    result_directory.mkdir(parents=True)

    return result_directory
