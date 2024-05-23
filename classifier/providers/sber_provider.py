import json
import os
import tiktoken
from gigachat import GigaChat
from gigachat.models.chat import Chat
from gigachat.models import ChatCompletion
from . import CompletionProvider, CompletionRequest, CompletionResponse
from classifier.templates import zero_shot_template
from .abstract_remote_execution_provider import AbstractRemoteExecutionCompletionProvider
from ..configuration import ConfigurationError
from ..logger import logger

models = ["GigaChat-preview", "GigaChat-Plus-preview", "GigaChat-Pro"]
GIGACHAT_API_KEY = os.environ.get("GIGACHAT_API_KEY")


# chat = GigaChat(credentials=GIGACHAT_API_KEY, model=models[0], verify_ssl_certs=False)


class SberCompletionProvider(CompletionProvider):
    __name: str
    __template = None
    __chat: GigaChat = None

    def __init__(self):
        if GIGACHAT_API_KEY is None:
            raise ValueError("Attempt to initialize a class without an access key.")

    @property
    def provider(self):
        return "sber"

    def configure(self, configuration: dict) -> None:
        self.__template = configuration.get("subject", self.__template)
        self.__name = str(configuration.get("name"))
        if configuration.get("engine") is not None:
            self.__chat = GigaChat(credentials=GIGACHAT_API_KEY,
                                   model=configuration.get("engine"),
                                   verify_ssl_certs=False)
        if self.__name is None:
            raise ConfigurationError("name")

    def get_completion(
            self, request: CompletionRequest, dry_run: bool = False
    ) -> CompletionResponse:
        if self.__chat is None:
            raise ValueError("Engine has not been configured.")
        if dry_run:
            return CompletionResponse(None, 0.)
        messages = [{
            "role": "system",
            "content": zero_shot_template[self.__template]
        }]

        for entry in request.samples:
            messages.append({
                "role": "user",
                "content": entry.input_text
            })
            messages.append({
                "role": "assistant",
                "content": entry.output_text
            })

        messages.append({
            "role": "user",
            "content": request.question
        })
        chat = Chat(
            messages=messages,
            stream=False,
            temperature=0.01,
        )
        res = self.__chat.chat(chat)
        return CompletionResponse(res.choices[0].message.content, 0)
