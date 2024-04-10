from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.Logger("classifier")


@dataclass
class LoggerConfiguration:

    filename: str = "classifier.log"
    enable_file_logging: bool = True
    enable_stdio_logging: bool = True
    file_log_level: int = logging.DEBUG
    stdio_log_level: int = logging.WARNING


def init_logger(config: LoggerConfiguration):

    if config.enable_file_logging:
        fh = logging.FileHandler(config.filename, encoding="utf-8")
        fh.setLevel(config.file_log_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(fh)

    if config.enable_stdio_logging:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(config.stdio_log_level)
        logger.addHandler(sh)
