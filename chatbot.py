from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = ".env"
if os.path.exists(env_path):
    try:
        load_dotenv(env_path)
    except Exception as e:
        print(f"Loading .env file manually (dotenv failed: {type(e).__name__})...")
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
            print("Successfully loaded .env file")
        except Exception as e2:
            print(f"Error reading .env file: {e2}")

# Login to Hugging Face (needed to download base model)
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
else:
    print("Warning: HF_TOKEN not found. You may need it to download the base model.")

# Model configuration
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./zeller-mistral"

# Check if CUDA is available
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Running on CPU (will be slow).")
    device_map = "cpu"
    quantization_config = None
else:
    device_map = "auto"
    # Use the same quantization as training
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)
print("This may take a few minutes on first run...")
print("=" * 60)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded!")

# Load base model with quantization
print("\nLoading base model with 4-bit quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    print("✓ Base model loaded!")
except Exception as e:
    print(f"\nError loading base model: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have internet connection (needs to download base model)")
    print("2. Check that HF_TOKEN is set in .env file")
    print("3. Ensure you have at least 6GB+ GPU memory available")
    raise

# Load LoRA adapter
print("\nLoading LoRA adapter...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✓ LoRA adapter loaded!")
except Exception as e:
    print(f"\nError loading LoRA adapter: {e}")
    print(f"Make sure the model was trained and saved to: {adapter_path}")
    raise

# Merge adapter for faster inference (optional - comment out if you want to keep adapter separate)
print("\nMerging LoRA adapter for faster inference...")
try:
    model = model.merge_and_unload()
    print("✓ Adapter merged!")
except Exception as e:
    print(f"Warning: Could not merge adapter: {e}")
    print("Continuing with adapter loaded (slightly slower but still works)")

model.eval()  # Set to evaluation mode
print("\n" + "=" * 60)
print("Model Ready!")
print("=" * 60)

# Chat function
def chat(user_input, conversation_history=None):
    """Generate a response from the model"""
    if conversation_history is None:
        conversation_history = []
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Format using chat template with full conversation history
    formatted = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Store input length to extract only new tokens later
    input_length = inputs['input_ids'].shape[1]
    
    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Reduced from 512 to prevent rambling
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Prevent repetition and rambling
        )
    
    # Extract only the new tokens (assistant's response)
    # Decode only the newly generated tokens, not the entire sequence
    new_tokens = outputs[0][input_length:]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up the response
    assistant_response = assistant_response.strip()
    
    # Remove EOS token if present at the start
    if tokenizer.eos_token and assistant_response.startswith(tokenizer.eos_token):
        assistant_response = assistant_response[len(tokenizer.eos_token):].strip()
    
    # Stop the response if it starts generating user input patterns (indicates it's going off-track)
    stop_patterns = ["\n\nUser:", "\n\nuser:", "User:", "user:", "\nUser:", "\nuser:"]
    for pattern in stop_patterns:
        if pattern in assistant_response:
            assistant_response = assistant_response.split(pattern)[0].strip()
            break
    
    # Also stop if it looks like it's starting a new conversation turn
    # Check for common patterns that indicate the model is continuing beyond its response
    if "\n\n" in assistant_response:
        # If there are multiple paragraphs, take only the first substantial one
        # This prevents the model from continuing to generate
        paragraphs = assistant_response.split("\n\n")
        if len(paragraphs) > 2:
            # Take first two paragraphs max, but only if they're substantial
            filtered = []
            for para in paragraphs[:2]:
                if len(para.strip()) > 10:  # Only keep substantial paragraphs
                    filtered.append(para)
            if filtered:
                assistant_response = "\n\n".join(filtered).strip()
    
    # Final cleanup - remove any trailing incomplete sentences or fragments
    # If the response ends mid-sentence with a very long response, truncate it
    if len(assistant_response) > 500:  # If response is very long
        # Try to find a good stopping point (end of sentence)
        last_period = assistant_response.rfind('.')
        last_exclamation = assistant_response.rfind('!')
        last_question = assistant_response.rfind('?')
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > 200:  # If we found a sentence end after a reasonable length
            assistant_response = assistant_response[:last_sentence_end + 1].strip()
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response, conversation_history

# Interactive chat loop
print("\n" + "=" * 60)
print("Chatbot Ready! Start chatting (type 'quit' or 'exit' to end)")
print("=" * 60)
print()

conversation_history = []

while True:
    try:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Generate response
        print("\nThinking...", end="", flush=True)
        response, conversation_history = chat(user_input, conversation_history)
        print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..." message
        print(f"ZellerBot: {response}\n")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nPlease try again or type 'quit' to exit.\n")
        # Reset conversation history on error to prevent corruption
        conversation_history = []

