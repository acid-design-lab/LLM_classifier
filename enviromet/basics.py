from pathlib import Path


def parse(content: str) -> dict:

    """This func parses config file and returns dict of initial experiment data"""

    lines = map(lambda x: x.split("=", 1), content.splitlines())
    data = {}
    for k, v in lines:
        data[k] = v.strip()

    data["classes"] = data.get("classes", "").split(",")
    data["is_text"] = data.get("data_format", "text") != "table"
    data["n_for_train"] = int(data.get("n_for_train"))
    data["test_size"] = float(data.get("test_size"))
    data["seed"] = int(data.get("seed"))

    return data


def create_result_directory(config: dict) -> str:

    """This func creates a directory in which your experiments results will be located."""

    experiments_directory = Path("./experiments")
    (experiments_directory / config["name"]).mkdir(parents=True, exist_ok=True)
    user_directory = (experiments_directory / config["name"])
    net_directory = user_directory / config["nnet"]
    dir_number = max(list(map(lambda x: int(x.stem), net_directory.glob("*"))) + [0])
    result_directory = net_directory / str(dir_number + 1)
    result_directory.mkdir(parents=True)

    return result_directory
