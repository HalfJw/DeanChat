from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
import torch

# --- 1. Load base model and tokenizer ---
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,  # saves VRAM if you have limited GPU
)

# --- 2. Apply LoRA (lightweight fine-tuning) ---
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- 3. Load and preprocess dataset ---
dataset = load_dataset("json", data_files="zeller.json")

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized = dataset.map(preprocess, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# --- 4. Training settings ---
args = TrainingArguments(
    output_dir="./zeller-mistral",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    report_to="none",
)

# --- 5. Train ---
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"])
trainer.train()

# --- 6. Save model ---
trainer.save_model("./zeller-mistral")
tokenizer.save_pretrained("./zeller-mistral")
