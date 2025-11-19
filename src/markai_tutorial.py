#! /usr/bin/env python3

import psutil
import time

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel


def clear_mps_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def process_batch_optimized(texts, model, tokenizer, batch_size=16):
    # Optimal batch processing for Mac M3
    device = next(model.parameters()).device
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors='pt', 
            max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results.extend(outputs.last_hidden_state.cpu().numpy())
        
        # Clear cache after each batch
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    return results


class TransformersManager:
    def __init__(self, model_name):
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def encode_text(self, text):
        inputs = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def monitor_resources(duration=60):
    print('Monitoring system resources...')
    
    for i in range(duration):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f'CPU: {cpu_percent}% | Memory: {memory.percent}% | '
              f'Available: {memory.available / (1024**3):.1f}GB')
        
        time.sleep(1)


def print_mps_memory():
    """
    # Use during model operations
    print_mps_memory()
    # ... run model inference ...
    print_mps_memory()
    """
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024**2)  # MB
        reserved = torch.mps.driver_allocated_memory() / (1024**2)   # MB
        
        print(
            f'MPS Memory - Allocated: {allocated:.1f}MB | Reserved: '
            '{reserved:.1f}MB')
    else:
        print('MPS not available')


def benchmark_model(model_name, texts, num_runs=5):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Warm up
    inputs = tokenizer(texts[0], return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _ = model(**inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model(**inputs)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f'Average time for {len(texts)} texts: {avg_time:.3f}s')
    print(f'Texts per second: {len(texts) / avg_time:.1f}')


def main():
    """
    From:
        https://markaicode.com/transformers-development-environment-mac-m3-setup/
    """

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('torch.device=', device)

    # torch.backends.mps.enable_empty_cache()

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    # sum(p.numel() for p in model.parameters())

    text = 'Meow said the cat and woof said the dog.'
    inputs = tokenizer(text, return_tensors='pt')

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    # output.last_hidden_state.shape

    '''
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export OMP_NUM_THREADS=8
    '''


    config = AutoConfig.from_pretrained(model_name)
    config.dtype = torch.float32  # MPS works best with float32

    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        dtype=torch.float32,
        device_map='auto')


if __name__ == '__main__':
    main()
