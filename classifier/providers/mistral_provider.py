from __future__ import annotations

import json

import tiktoken
from openai.types.chat import ChatCompletion

from . import CompletionRequest
from . import CompletionResponse
from .abstract_remote_execution_provider import (
    AbstractRemoteExecutionCompletionProvider,
)
from classifier.templates import zero_shot_template


class MistralCompletionProvider(AbstractRemoteExecutionCompletionProvider):
    __name: str
    __template = None

    @property
    def provider(self):
        return "mistral"

    def configure(self, configuration: dict) -> None:
        self.__template = configuration.get("subject", self.__template)
        AbstractRemoteExecutionCompletionProvider.configure(self, configuration)

    def _to_request_data(self, request: CompletionRequest) -> dict:

        if self.__template is None:
            raise ValueError("Enable to create request: template has not specified")
        try:
            content = [{"role": "user", "content": zero_shot_template[self.__template]}]
        except KeyError:
            raise ValueError(f"Enable to create request: \"template\" configuration parameter is invalid. Valid "
                             f"options are: {', '.join(zero_shot_template.keys())}")

        for entry in request.samples:
            content.append({"role": "user", "content": entry.input_text})
            content.append({"role": "assistant", "content": entry.output_text})

        return {
            "question": request.question,
            "name": "",
            "request": content,
            "engine": request.engine,
        }

    def _estimate_cost(self, request: CompletionRequest) -> float:

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(
            json.dumps(
                self._to_request_data(request)["request"]
                + [{"role": "user", "content": request.question}],
                ensure_ascii=False,
            )
        )
        return self._prompt_tokens_to_price(len(tokens), request.engine)

    @staticmethod
    def _prompt_tokens_to_price(tokens: int, engine: str) -> float:

        match engine:
            case "mistral-medium-latest":
                return 2.7 / 1_000_000 * tokens
            case "mistral-small-latest":
                return 2 / 1_000_000 * tokens
            case "mistral-large-latest":
                return 8 / 1_000_000 * tokens

            case _:
                raise ValueError("Invalid engine!")

    @staticmethod
    def _completion_tokens_to_price(tokens: int, engine: str) -> float:

        match engine:
            case "mistral-medium-latest":
                return 8.1 / 1_000_000 * tokens
            case "mistral-small-latest":
                return 6 / 1_000_000 * tokens
            case "mistral-large-latest":
                return 24 / 1_000_000 * tokens
            case _:
                raise ValueError("Invalid engine!")

    def _from_response_data(
        self, request: CompletionRequest, response: dict
    ) -> CompletionResponse:

        model = ChatCompletion.model_validate(response)
        if not model.choices[0].message.content:
            raise ValueError("Model returned an invalid response")
        return CompletionResponse(
            text=model.choices[0].message.content,
            cost=self._completion_tokens_to_price(
                model.usage.completion_tokens, request.engine
            )
            + self._prompt_tokens_to_price(model.usage.prompt_tokens, request.engine),
        )
