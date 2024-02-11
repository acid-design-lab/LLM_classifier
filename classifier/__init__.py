from __future__ import annotations

import logging

from classifier.logger import init_logger
from classifier.logger import LoggerConfiguration

logger_configuration = LoggerConfiguration()
logger_configuration.stdio_log_level = logging.INFO
init_logger(logger_configuration)
