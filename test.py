from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Handle potential encoding issues with .env file
try:
    load_dotenv()
except (UnicodeDecodeError, Exception) as e:
    # If .env file has encoding issues, try to read it manually
    print(f"Warning: Issue loading .env file ({type(e).__name__}). Trying to read manually...")
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            # Use utf-8-sig to automatically strip BOM if present
            with open(env_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as read_error:
            print(f"Could not read .env file: {read_error}")

# Log into Hugging Face using token from environment variable
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable not found. "
        "Please create a .env file with your Hugging Face token in UTF-8 encoding. "
        "See .env.example for reference."
    )
login(hf_token)

# Model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model efficiently
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                # Automatically places layers on GPU(s) or CPU
    torch_dtype=torch.float16,        # Half precision to save VRAM (use bfloat16 on A100/H100)
    low_cpu_mem_usage=True            # Reduces CPU memory load during loading
)

# Optional: Enable efficient attention (if available)
if hasattr(model, "enable_flash_attn"):
    model.enable_flash_attn()  # Faster attention if supported

# Prompt
prompt = "Explain recursion in simple terms."

# Tokenize and move to device
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.inference_mode():  # Avoid gradient tracking for inference
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,      # Add some creativity
        top_p=0.9,            # Nucleus sampling
        do_sample=True        # Randomized generation
    )

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

