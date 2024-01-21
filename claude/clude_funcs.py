from __future__ import annotations

import requests


def initialize(api_key):
    """Before calling any endpoints, we need an HTTP client in our chosen programming language to send requests and
    receive responses."""

    headers = {"Authorization": f"{api_key}"}
    client = requests.Session()
    client.headers.update(headers)
    return client


def conversation(client, prompt, new_conversation=True):
    """The conversations endpoint is perhaps the most versatile of all the Claude 2 APIs. As the name suggests,
    it allows free-flowing dialogue with multiple exchanges.
    We can now send followup questions, getting responses that build on the existing context"""

    if new_conversation:
        prompt = (
            "I'm planning a trip to Paris next spring. Any recommendations on where I should visit or what I "
            "should be sure to do?"
        )
        response = client.post(
            "https://api.anthropic.com/v2/conversations",
            json={"messages": [{"content": prompt}]},
        )
        return response.json()
    else:
        followup = (
            "I'm a big art lover - what museum would you recommend as a must-see?"
        )
        response = client.post(
            "https://api.anthropic.com/v2/conversations",
            json={"messages": [{"content": followup}]},
        )
        return response.json()


def completions(client, prompt):
    """f you want Claude 2 to expand on some initial text with relevant, high quality continuations, the completions
    endpoint is perfect."""

    prompt = (
        "Meeting notes\nPresent: Bob, Susan and Joan\nAgenda:\n- Employee engagement survey results\n- New "
        "health insurance provider"
    )
    response = client.post(
        "https://api.anthropic.com/v2/completions", json={"prompt": prompt}
    )
    return response.json()


def answer(client, quastion):
    """When you have a specific question, use the answers endpoint to get Claude’s direct response."""

    question = "What year did the first airplane fly?"
    response = client.post(
        "https://api.anthropic.com/v2/answers", json={"question": question}
    )
    return response.json()


def search(client, query):
    """When researching a particular topic, Claude 2’s search endpoint finds the most relevant excerpt text from its
    broad collection knowledge."""

    query = "text on best practices for watering a vegetable garden"
    response = client.post("https://api.anthropic.com/v2/search", json={"query": query})
    return response.json()


def classifications(client, content):
    """The classifications endpoint applies one of Claude 2’s trained models to input text for analysis."""

    content = "Your receipt from Amazon Store Card & Special Financing on May 24..."
    response = client.post(
        "https://api.anthropic.com/v2/classifications",
        json={"model": "impersonation-detection", "content": content},
    )
    return response.json()


def embedding(client, text):
    """The final Claude 2 endpoint we’ll cover generates vector embeddings – mathematical representations of text
    meaning."""

    text = "Claude 2 is the latest conversational AI assistant created by Anthropic."
    response = client.post(
        "https://api.anthropic.com/v2/embeddings", json={"content": text}
    )
    return response.json()
