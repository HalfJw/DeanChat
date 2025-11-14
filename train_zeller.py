from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import json
import os
import random
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
# Handle potential encoding issues with .env file
env_path = ".env"
if os.path.exists(env_path):
    # Try to load with python-dotenv first
    try:
        load_dotenv(env_path)
    except Exception as e:
        # If dotenv fails, read manually with UTF-8 encoding
        print(f"Loading .env file manually (dotenv failed: {type(e).__name__})...")
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
            print("Successfully loaded .env file")
        except UnicodeDecodeError:
            # Try with utf-8-sig (handles BOM)
            try:
                with open(env_path, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            value = value.strip().strip('"').strip("'")
                            os.environ[key.strip()] = value
                print("Successfully loaded .env file (UTF-8 with BOM)")
            except Exception as e2:
                print(f"Error reading .env file: {e2}")
                print("Please ensure your .env file is saved in UTF-8 encoding.")
                raise
        except Exception as e2:
            print(f"Error reading .env file: {e2}")
            raise
else:
    print(f"Warning: .env file not found at {os.path.abspath(env_path)}")
    print("Please create a .env file with your Hugging Face token.")

# Login to Hugging Face using token from environment variable
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable not found. "
        "Please create a .env file with your Hugging Face token in UTF-8 encoding. "
        "See .env.example for reference."
    )
login(hf_token)

# --- 1. Load base model and tokenizer ---
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Check if CUDA is available (required for 4-bit quantization)
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires a CUDA-capable GPU.\n"
        "Please ensure:\n"
        "1. You have an NVIDIA GPU installed\n"
        "2. CUDA is properly installed\n"
        "3. PyTorch is installed with CUDA support\n"
        "\nTo install PyTorch with CUDA, visit: https://pytorch.org/get-started/locally/"
    )

# Display GPU information
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
cuda_version = torch.version.cuda

print("=" * 60)
print("GPU Setup")
print("=" * 60)
print(f"GPU: {gpu_name}")
print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
print(f"CUDA Version: {cuda_version}")
print(f"PyTorch Version: {torch.__version__}")
print("=" * 60)

# Check if GPU has enough memory (4-bit quantized model needs ~4-6GB)
if gpu_memory_gb < 6:
    print(f"Warning: GPU has {gpu_memory_gb:.2f} GB memory.")
    print("Training may fail with out-of-memory errors.")
    print("Consider reducing batch size if you encounter memory issues.")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully!")

# Configure 4-bit quantization for GPU
print("\nConfiguring 4-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normalized Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
    bnb_4bit_use_double_quant=True,  # Enable double quantization for better memory efficiency
)

# Load model with quantization
print("Loading model with 4-bit quantization (this may take a few minutes)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distribute model across available GPUs
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"\nError loading model: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have at least 6GB+ GPU memory available")
    print("2. Close other applications using GPU memory")
    print("3. Try reducing the batch size in TrainingArguments")
    print("4. Check that bitsandbytes is properly installed:")
    print("   pip install bitsandbytes")
    raise

# --- 2. Prepare model for training ---
print("\nPreparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("Model prepared for training!")

# --- 3. Apply LoRA (lightweight fine-tuning) ---
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- 3. Load and preprocess dataset ---
dataset = load_dataset("json", data_files="zeller.json")

def format_instruction(text):
    """Format text as an instruction-response pair for chatbot training"""
    # Create varied instructions that would prompt this response
    # This helps the model learn to respond to different question styles
    instructions = [
        "Can you explain this?",
        "What should I know about this?",
        "Tell me about this topic.",
        "How does this work?",
        "What are the requirements?",
        "Can you provide more information?",
        "I have a question about this.",
        "What do I need to know?",
        "Explain this to me.",
        "Help me understand this.",
    ]
    
    # Randomly select an instruction format for variety
    instruction = random.choice(instructions)
    
    # Format using Mistral's chat template
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": text}
    ]
    
    # Use the tokenizer's chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted

def preprocess_function(examples):
    """Tokenize and format the examples"""
    # Format each text as an instruction-response pair
    texts = []
    for text in examples["text"]:
        formatted = format_instruction(text)
        texts.append(formatted)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Preprocessing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# --- 5. Data collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# --- 6. Training settings ---
print("\nConfiguring training arguments...")
args = TrainingArguments(
    output_dir="./zeller-mistral",
    per_device_train_batch_size=2,  # Adjust based on GPU memory (reduce if OOM)
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_strategy="steps",
    fp16=True,  # Use mixed precision for faster training
    bf16=False,  # Set to True if you have Ampere+ GPU (RTX 30xx, A100, etc.)
    learning_rate=2e-4,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    report_to="none",
    warmup_steps=100,
    max_steps=-1,  # Use num_train_epochs instead
    remove_unused_columns=False,  # Keep all columns
    dataloader_pin_memory=True,  # Speed up data loading
    gradient_checkpointing=True,  # Save memory at cost of speed
)
print("Training arguments configured!")

# --- 7. Train ---
print("Starting training...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# --- 6. Save model ---
trainer.save_model("./zeller-mistral")
tokenizer.save_pretrained("./zeller-mistral")
