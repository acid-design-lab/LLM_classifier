from __future__ import annotations

import json
import os
import random

import requests
from openai.types.chat.chat_completion import ChatCompletion

from gpt.templates import zero_shot_template


def create_request_template(
    data: list[(str, str)],
    name: str,
    seed: int,
) -> list:
    """This func generates request template for selected model to classify text using few-shots method."""

    request = [{"role": "system", "content": zero_shot_template["chemistry"]}]

    random.seed(seed)

    for entry in data:
        request.append({"role": "user", "name": name, "content": entry[0]})
        request.append({"role": "assistant", "content": entry[1]})

    return request


def request_completion(
    question: str, name: str, request: list[dict[str, str]]
) -> ChatCompletion:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")
    return ChatCompletion.model_validate(
        requests.get(
            API_URL + "/respond",
            data=json.dumps({"question": question, "name": name, "request": request}),
        ).json()
    )
