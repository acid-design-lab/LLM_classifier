from __future__ import annotations

import abc
import json
import math
import os
from pprint import pprint
from typing import Any

import requests

from . import CompletionProvider
from . import CompletionRequest
from . import CompletionResponse
from classifier.configuration import ConfigurationError
from classifier.logger import logger


class AbstractRemoteExecutionCompletionProvider(CompletionProvider, abc.ABC):
    def __init__(self):
        self.__retry_number = 5
        self._API_URL = None

    @property
    @abc.abstractmethod
    def provider(self):
        pass

    def configure(self, configuration: dict) -> None:
        self._API_URL = (
            configuration.get("api_url") or self._API_URL or os.environ.get("API_URL")
        )
        if self._API_URL is None:
            raise ConfigurationError("api_url")
        if not self._API_URL.startswith("http"):
            self._API_URL = "http://" + self._API_URL
        if not self._API_URL.endswith("/"):
            self._API_URL += "/"
        tmp = configuration.get("retry_number")
        if tmp is not None:
            self.__retry_number = int(tmp)

    @abc.abstractmethod
    def _to_request_data(self, request: CompletionRequest) -> dict:
        """Create dict that represents the request out of CompletionRequest object"""
        pass

    @abc.abstractmethod
    def _from_response_data(
        self, request: CompletionRequest, response: dict
    ) -> CompletionResponse:
        """Create CompletionResponse object from json data of the API response."""
        pass

    @abc.abstractmethod
    def _estimate_cost(self, request: CompletionRequest) -> float:
        """Estimates the cost of the specific request"""
        pass

    def get_completion(
        self, request: CompletionRequest, dry_run: bool = False
    ) -> CompletionResponse:
        request_dict: dict[str, Any] = self._to_request_data(request)
        err: requests.ConnectionError | None = None

        if dry_run:
            return CompletionResponse(
                text=None, cost=math.ceil(self._estimate_cost(request) * 100) / 100
            )
        for n in range(self.__retry_number):
            try:
                response = requests.get(
                    self._API_URL + "respond", data=json.dumps(request_dict)
                )
                result = self._from_response_data(request, response.json())
            except requests.ConnectionError as e:
                logger.warning("Connection failed. Retrying...")
                err = e
                continue
            else:
                return result
        if err:
            raise err
        else:
            raise Exception("No response")
