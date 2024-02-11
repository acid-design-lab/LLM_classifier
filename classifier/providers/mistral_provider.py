from __future__ import annotations

from openai.types.chat import ChatCompletion

from . import CompletionRequest
from . import CompletionResponse
from .abstract_remote_execution_provider import (
    AbstractRemoteExecutionCompletionProvider,
)
from classifier.templates import zero_shot_template


class MistralCompletionProvider(AbstractRemoteExecutionCompletionProvider):
    __name: str

    @property
    def provider(self):
        return "mistral"

    def configure(self, configuration: dict) -> None:
        AbstractRemoteExecutionCompletionProvider.configure(self, configuration)

    def _to_request_data(self, request: CompletionRequest) -> dict:
        content = [{"role": "user", "content": zero_shot_template["chemistry"]}]
        for entry in request.samples:
            content.append({"role": "user", "content": entry.input_text})
            content.append({"role": "assistant", "content": entry.output_text})
        return {
            "question": request.question,
            "name": self.__name,
            "request": content,
            "engine": request.engine,
        }

    def _estimate_cost(self, request: CompletionRequest) -> float:
        tokens_len = 0
        return self._prompt_tokens_to_price(tokens_len, request.engine)

    @staticmethod
    def _prompt_tokens_to_price(tokens: int, engine: str) -> float:
        match engine:
            case "mistral-medium":
                return 2.5 / 1.08 / 1_000_000 * tokens
            case "mistral-small":
                return 0.6 / 1.08 / 1_000_000 * tokens
            case "mistral-tiny":
                return 0.14 / 1.08 / 1_000_000 * tokens
            case _:
                raise ValueError("Invalid engine!")

    @staticmethod
    def _completion_tokens_to_price(tokens: int, engine: str) -> float:
        match engine:
            case "mistral-medium":
                return 7.5 / 1.08 / 1_000_000 * tokens
            case "mistral-small":
                return 1.8 / 1.08 / 1_000_000 * tokens
            case "mistral-tiny":
                return 0.42 / 1.08 / 1_000_000 * tokens
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
