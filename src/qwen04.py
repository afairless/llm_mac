
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import anyio
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# Load model and tokenizer
# ----------------------
model_name = 'Qwen/Qwen2.5-Coder-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# ----------------------
# Token stream generator
# ----------------------

def generate_stream(inputs, max_tokens=500):
    """
    Synchronous generator to stream token-by-token decoded text.
    """
    # Example: simple concatenation of all messages
    prompt = '\n'.join([msg['content'] for msg in inputs])

    # Encode prompt to model input IDs
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)

    # Generate tokens
    output_ids = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens)

    # Stream token by token
    for token_id in output_ids[0]:
        # Decode a single token (as a list) to avoid TypeError
        decoded = tokenizer.decode([token_id.item()], skip_special_tokens=True)
        if decoded.strip():  # skip empty tokens
            yield decoded

# ----------------------
# Async streaming endpoint
# ----------------------
@app.post('/v1/chat/completions/stream')
async def stream_endpoint(request: Request):
    data = await request.json()
    inputs = data.get('messages', [])
    max_tokens = data.get('max_tokens', 500)

    async def async_gen():
        # Run blocking generator in a thread to avoid blocking the event loop
        for chunk in await anyio.to_thread.run_sync(
            lambda: list(generate_stream(inputs, max_tokens))):
            yield chunk.encode('utf-8')

    return StreamingResponse(async_gen(), media_type='text/plain')

# ----------------------
# Test endpoint
# ----------------------
@app.get('/')
def root():
    return {'message': 'Qwen streaming API is running!'}


