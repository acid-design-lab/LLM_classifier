from __future__ import annotations

import json

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

    @property
    def provider(self):
        return "anthropic"

    def configure(self, configuration: dict) -> None:
        AbstractRemoteExecutionCompletionProvider.configure(self, configuration)
        self.__name = str(configuration.get("name"))
        if self.__name is None:
            raise ConfigurationError("name")

    def _to_request_data(self, request: CompletionRequest) -> dict:
        content = [{"role": "system", "content": zero_shot_template["chemistry"]}]

        for entry in request.samples:
            content.append(
                {"role": "user", "content": entry.input_text}
            )
            content.append({"role": "assistant", "content": entry.output_text})

        return {
            "question": request.question,
            "request": content,
            "engine": request.engine,
            "name": ""
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
