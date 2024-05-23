from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable

import magic
import tiktoken

from . import CompletionRequest
from . import CompletionResponse
from ..logger import logger
from .abstract_remote_execution_provider import (
    AbstractRemoteExecutionCompletionProvider,
)
from classifier.configuration import ConfigurationError
from classifier.templates import zero_shot_template

from anthropic.types import Message


class AnthropicCompletionProvider(AbstractRemoteExecutionCompletionProvider):
    __name: str
    __template = None
    __store_fn: Callable[[str], str] = None
    __convert_fn: Callable[[str], str] = None
    __vision: bool = False

    @property
    def provider(self):
        return "anthropic"

    def configure(self, configuration: dict) -> None:
        AbstractRemoteExecutionCompletionProvider.configure(self, configuration)
        self.__template = configuration.get("subject", self.__template)
        self.__convert_fn = configuration.get("convert_fn", self.__convert_fn)
        self.__vision = configuration.get("vision", self.__vision)
        self.__name = str(configuration.get("name"))
        if self.__name is None:
            raise ConfigurationError("name")

    def _to_request_data(self, request: CompletionRequest) -> dict:
        if self.__template is None:
            raise ValueError("Enable to create request: template has not specified")
        try:
            content = [{"role": "system", "content": zero_shot_template[self.__template]}]
        except KeyError:
            raise ValueError(f"Enable to create request: \"template\" configuration parameter is invalid. Valid "
                             f"options are: {', '.join(zero_shot_template.keys())}")
        if self.__vision:
            for entry in request.samples:
                filepath = Path(self.__convert_fn(entry.input_text.removeprefix("smiles: ")))
                content.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": (base64.b64encode(filepath.read_bytes())).decode("utf-8")
                        }}
                    ]
                })
                content.append({"role": "assistant", "content": entry.output_text})
            filepath = Path(self.__convert_fn(request.question.removeprefix("smiles: ")))
            return {
                "question": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": (base64.b64encode(filepath.read_bytes())).decode("utf-8")
                    }}
                ],
                "name": "",
                "request": content,
                "engine": request.engine,
            }
        for entry in request.samples:
            content.append(
                {"role": "user", "content": entry.input_text}
            )
            content.append({"role": "assistant", "content": entry.output_text})

        return {
            "question": request.question,
            "name": "",
            "request": content,
            "engine": request.engine,
        }


    # def _to_request_data(self, request: CompletionRequest) -> dict:
    #     if self.__template is None:
    #         raise ValueError("Enable to create request: template has not specified")
    #     try:
    #         content = [{"role": "system", "content": zero_shot_template[self.__template]}]
    #     except KeyError:
    #         raise ValueError(f"Enable to create request: \"template\" configuration parameter is invalid. Valid "
    #                          f"options are: {', '.join(zero_shot_template.keys())}")
    #
    #     for entry in request.samples:
    #         content.append(
    #             {"role": "user", "content": entry.input_text}
    #         )
    #         content.append({"role": "assistant", "content": entry.output_text})
    #
    #     return {
    #         "question": request.question,
    #         "request": content,
    #         "engine": request.engine,
    #         "name": ""
    #     }

    def _estimate_cost(self, request: CompletionRequest) -> float:
        enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(
            json.dumps(
                self._to_request_data(request)["request"]
                + [{"role": "user", "name": self.__name, "content": request.question}],
                ensure_ascii=False,
            )
        )
        return self._prompt_tokens_to_price(len(tokens), request.engine)

    @staticmethod
    def _prompt_tokens_to_price(tokens: int, engine: str) -> float:
        match engine:
            case "claude-3-haiku-20240307":
                return (0.25 / 1000_000) * tokens
            case "claude-3-sonnet-20240229":
                return (3. / 1000_000) * tokens
            case "claude-3-opus-20240229":
                return (15. / 1000_000) * tokens
            case _:
                raise ValueError(f'Invalid engine "{engine}"')

    @staticmethod
    def _completion_tokens_to_price(tokens: int, engine: str) -> float:
        match engine:
            case "claude-3-haiku-20240307":
                return (1.25 / 1000_000) * tokens
            case "claude-3-sonnet-20240229":
                return (15. / 1000_000) * tokens
            case "claude-3-opus-20240229":
                return (75. / 1000_000) * tokens
            case _:
                raise ValueError(f'Invalid engine "{engine}"')

    def _from_response_data(
            self, request: CompletionRequest, response: dict
    ) -> CompletionResponse:
        # print(response)
        model = Message.model_validate(response)
        if not model.content:
            raise ValueError("Model returned an invalid response")
        content = ''.join(p.text for p in model.content)
        price = self._prompt_tokens_to_price(
            model.usage.input_tokens, request.engine
        ) + self._completion_tokens_to_price(
            model.usage.output_tokens, request.engine
        )

        new_response = CompletionResponse(
            text=content, cost=price
        )

        logger.debug(
            f"Anthropic reported {model.usage.input_tokens} prompt tokens and {model.usage.output_tokens} "
            f"completion ({price:.4f}$) tokens. Estimation was: {self._estimate_cost(request):.4f}$"
        )

        return new_response
