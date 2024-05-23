from __future__ import annotations
import json
from pprint import pprint
from typing import Callable
import tiktoken
from openai.types.chat.chat_completion import ChatCompletion
from . import CompletionRequest
from . import CompletionResponse
from ..logger import logger
from .abstract_remote_execution_provider import (
    AbstractRemoteExecutionCompletionProvider,
)
from classifier.configuration import ConfigurationError
from classifier.templates import zero_shot_template


class OpenAICompletionProvider(AbstractRemoteExecutionCompletionProvider):
    __name: str
    __template = None
    __store_fn: Callable[[str], str] = None
    __convert_fn: Callable[[str], str] = None
    __vision: bool = False

    @property
    def provider(self):
        return "openai"

    def configure(self, configuration: dict) -> None:
        AbstractRemoteExecutionCompletionProvider.configure(self, configuration)
        self.__template = configuration.get("subject", self.__template)
        self.__store_fn = configuration.get("store_fn", self.__store_fn)
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
                content.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": self.__store_fn(self.__convert_fn(entry.input_text.removeprefix("smiles: ")))
                        }}
                    ]
                })
                content.append({"role": "assistant", "content": entry.output_text})
            return {
                "question": [
                    {"type": "image_url", "image_url": {
                        "url": self.__store_fn(self.__convert_fn(request.question.removeprefix("smiles: ")))
                    }}
                ],
                "name": self.__name,
                "request": content,
                "engine": request.engine,
            }
        for entry in request.samples:
            content.append(
                {"role": "user", "name": self.__name, "content": entry.input_text}
            )
            content.append({"role": "assistant", "content": entry.output_text})

        return {
            "question": request.question,
            "name": self.__name,
            "request": content,
            "engine": request.engine,
        }        

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
            case "gpt-4-turbo-preview":
                return (0.01 / 1000) * tokens
            case "gpt-4":
                return (0.03 / 1000) * tokens
            case "gpt-3.5-turbo-1106":
                return (0.0005 / 1000) * tokens
            case "gpt-3.5-turbo":
                return (0.0005 / 1000) * tokens
            case "gpt-4o":
                return (0.005 / 1000) * tokens
            case _:
                raise ValueError(f'Invalid engine "{engine}"')

    @staticmethod
    def _completion_tokens_to_price(tokens: int, engine: str) -> float:
        match engine:
            case "gpt-4-turbo-preview":
                return (0.03 / 1000) * tokens
            case "gpt-4":
                return (0.06 / 1000) * tokens
            case "gpt-3.5-turbo-1106":
                return (0.0015 / 1000) * tokens
            case "gpt-3.5-turbo":
                return (0.0015 / 1000) * tokens
            case "gpt-4o":
                return (0.0015 / 1000) * tokens
            case _:
                raise ValueError(f'Invalid engine "{engine}"')

    def _from_response_data(
        self, request: CompletionRequest, response: dict
    ) -> CompletionResponse:
        model = ChatCompletion.model_validate(response)
        if not model.choices[0].message.content:
            raise ValueError("Model returned an invalid response")

        price = self._prompt_tokens_to_price(
            model.usage.prompt_tokens, request.engine
        ) + self._completion_tokens_to_price(
            model.usage.completion_tokens, request.engine
        )

        new_response = CompletionResponse(
            text=model.choices[0].message.content, cost=price
        )

        logger.debug(
            f"OpenAI reported {model.usage.prompt_tokens} prompt tokens and {model.usage.completion_tokens} "
            f"completion ({price:.4f}$) tokens. Estimation was: {self._estimate_cost(request):.4f}$"
        )

        return new_response
