
from pathlib import Path
import requests
import json
from pydantic import BaseModel, Field, ValidationError
# import ollama
from openai import OpenAI
import outlines

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal, List, Any
from enum import Enum


def example01():

    url = 'http://localhost:11434/api/generate'

    data = {
        'model': 'gemma3',
        'prompt': 'Write a short haiku about Python programming.'}

    response = requests.post(url, json=data, stream=False)
    response.text

    for line in response.iter_lines():
        if line:
            message = json.loads(line)
            print(message.get('response', ''), end='')


def example02():
    """
    Example from:
        https://github.com/dottxt-ai/outlines
    """

    MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto'),
        AutoTokenizer.from_pretrained(MODEL_NAME))

    sentiment = model(
        "Analyze: 'This product completely changed my life!'",
        Literal['Positive', 'Negative', 'Neutral'])
    print(sentiment)  # 'Positive'

    temperature = model("What's the boiling point of water in Celsius?", int)
    print(temperature)  # 100

    class Rating(Enum):
        poor = 1
        fair = 2
        good = 3
        excellent = 4

    class ProductReview(BaseModel):
        rating: Rating
        pros: list[str]
        cons: list[str]
        summary: str

    review_str = (
        'Review: The XPS 13 has great battery life and a stunning display, but '
        'it runs hot and the webcam is poor quality.')
    review = model(
        review_str,
        ProductReview,
        max_new_tokens=200)

    review = ProductReview.model_validate_json(review)
    print(f'Rating: {review.rating.name}')  # 'Rating: good'
    print(f'Pros: {review.pros}')           # 'Pros: ['great battery life', 'stunning display']'
    print(f'Summary: {review.summary}')     # 'Summary: Good laptop with great display but thermal issues'


def example03():
    """
    Example from:
        https://dottxt-ai.github.io/outlines/latest/features/models/openai/
    """

    input_filepath = Path.home() / 'openai_key.txt'
    key = read_text_file(input_filepath, return_string=True)
    assert isinstance(key, str)

    MODEL_NAME = 'gpt-4o'
    client = OpenAI(api_key=key)
    model = outlines.from_openai(client, MODEL_NAME)

    response = model('What is the capital of Tennessee?', max_tokens=20)

    class Character(BaseModel):
        name: str
        age: int
        skills: List[str]

    # model = outlines.from_openai(openai.OpenAI(), 'gpt-4o')

    result = model('Create a character, use the json format.', Character, top_p=0.1)
    assert isinstance(result, str)
    print(result) # '{'name': 'Evelyn', 'age': 34, 'skills': ['archery', 'stealth', 'alchemy']}'
    print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']
    result_dict = json.loads(result)


if __name__ == '__main__':
    example01()
