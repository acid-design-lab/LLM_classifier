from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from gpt.templates import zero_shot_template
import random

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium"


# client = MistralClient(api_key=api_key)
#
# messages = [
#     ChatMessage(role="system", content="You are laconic."),
#     ChatMessage(role="user", content="What is the love in two words?")
# ]
#
# # No streaming
# chat_response = client.chat(
#     model=model,
#     messages=messages,
#     temperature=0.01,
#     # max_tokens=20,
#     top_p=0.01,
#     random_seed=2201,
# )
# print(chat_response.choices[0].message.content)


def create_request_template_mistral(data: list[(str, str)],
                                    seed: int, ) -> list:
    """This func generates request template for selected model to classify text using few-shots method."""

    # request = [ChatMessage(role="user", content=zero_shot_template["chemistry"])]
    request = [{"role": "user", "content": zero_shot_template["chemistry"]}]

    random.seed(seed)

    for entry in data:
        # request.append([ChatMessage(role="user", content=entry[0])])
        # request.append([ChatMessage(role="assistant", content=entry[1])])
        request.append({"role": "user", "content": entry[0]})
        request.append({"role": "assistant", "content": entry[1]})
    return request
