from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "./zeller-mistral"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

prompt = "Explain recursion in your teaching style."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
