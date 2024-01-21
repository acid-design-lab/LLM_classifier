import pathlib
import textwrap

import google.generativeai as genai


from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is the meaning of life?")
