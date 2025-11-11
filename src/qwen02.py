
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'Qwen/Qwen2.5-Coder-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='auto')

# Chat loop
system_prompt = 'You are a helpful coding assistant.'
chat_history = []

print(f'{model_name} chat. Type "exit" to quit.\n')

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break

    # Prepare the conversation
    messages = (
        [{'role': 'system', 'content': system_prompt}] + 
        chat_history + 
        [{'role': 'user', 'content': user_input}])

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f'Assistant: {response}\n')
    chat_history.append({'role': 'user', 'content': user_input})
    chat_history.append({'role': 'assistant', 'content': response})



