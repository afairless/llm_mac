
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = 'Qwen/Qwen2.5-Coder-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map='auto')

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = 512

@app.post('/v1/chat/completions')
def chat(request: ChatRequest):
    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in request.messages],
        tokenize=False,
        add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=request.max_tokens)
    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {
        'choices': [{'message': {'role': 'assistant', 'content': response}}]}


