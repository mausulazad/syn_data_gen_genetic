import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model card with names and model IDs
model_card = [
    # {"name": "slm", "model_id": "meta-llama/Llama-3.2-3B-Instruct"},
    {"name": "llama_32", "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct"},
    {"name": "llava_next", "model_id": "llava-hf/llava-v1.6-mistral-7b-hf"},
    {"name": "molmo", "model_id": "allenai/Molmo-7B-D-0924"},
    {"name": "llava_critic", "model_id": "lmms-lab/llava-critic-7b"},
    # {"name": "prometheus_vision", "model_id": "kaist-ai/prometheus-vision-13b-v1.0"},
    # {"name": "qwen2_vl", "model_id": "Qwen/Qwen2-VL-7B-Instruct"},
    # {"name": "qwen_25", "model_id": "Qwen/Qwen2.5-7B-Instruct"},
]

# Get available GPU count
num_gpus = torch.cuda.device_count()
assert num_gpus >= 4, f"Expected at least 4 GPUs, but found {num_gpus}"

# Assign models to GPUs in a round-robin manner
for i, model in enumerate(model_card):
    model["device"] = f"cuda:{i % num_gpus}"

# Load models
models = {}
for model_info in model_card:
    name, model_id, device = model_info["name"], model_info["model_id"], model_info["device"]
    print(f"Loading {name} on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    models[name] = {"model": model, "tokenizer": tokenizer}

print("All models loaded successfully.")
