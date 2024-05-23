import os
import time
from pprint import pprint
from . import CompletionProvider, CompletionRequest, CompletionResponse
import requests
from classifier.templates import zero_shot_template

KEY = os.environ.get("YANDEXGPT_API_KEY")
CATALOG = os.environ.get('YC_CATALOG')


class YandexGPTCompletionProvider(CompletionProvider):
    __template = None

    def configure(self, configuration: dict) -> None:
        self.__template = configuration.get("subject", self.__template)

    @property
    def provider(self):
        return "yandex"

    def get_completion(self, request: CompletionRequest, dry_run: bool = False) -> CompletionResponse:
        if dry_run:
            return CompletionResponse(None, 0.)
        messages = [{
            "role": "system",
            "text": zero_shot_template[self.__template]
        }]

        for entry in request.samples:
            messages.append({
                "role": "user",
                "text": entry.input_text
            })
            messages.append({
                "role": "assistant",
                "text": entry.output_text
            })

        messages.append({
            "role": "user",
            "text": request.question
        })

        body = {
            "modelUri": f"gpt://{CATALOG}/{request.engine}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": 8192
            },
            "messages": messages
        }
        headers = {
            "Authorization": f"Api-Key {KEY}",
            "x-folder-id": CATALOG
        }
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync", json=body,
                            headers=headers).json()

        while not res["done"]:
            res = requests.get(f"https://operation.api.cloud.yandex.net/operations/{res['id']}", headers=headers)
            res = res.json()
            time.sleep(0.2)
        if not res.get("response", None):
            pprint(res)
            input()
        res = res["response"]

        return CompletionResponse(res["alternatives"][0]["message"]["text"].lower(), (
                    int(res["usage"]["inputTextTokens"]) + int(res["usage"]["completionTokens"])) * 1.2 / 1000)
