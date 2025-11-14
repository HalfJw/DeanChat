# Teacher Chatbot

Train a free AI chatbot that sounds like your teacher using Mistral-7B-Instruct and LoRA fine-tuning.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Hugging Face API token:**
   - Create a `.env` file in the project root directory
   - Add your Hugging Face token to the file:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
   - Get your token from: https://huggingface.co/settings/tokens
   - **Important:** The `.env` file is ignored by git to keep your token private

3. **Prepare your data:**
   - Your JSON file (`zeller.json`) should have entries with a `"text"` field
   - Each entry should contain text from your teacher

4. **Train the model:**
   ```bash
   python train_zeller.py
   ```
   
   This will:
   - Load Mistral-7B-Instruct-v0.3 (free, open-source model)
   - Apply LoRA fine-tuning (efficient, requires less memory)
   - Train on your teacher's text data
   - Save the model to `./zeller-mistral`

5. **Chat with your AI teacher:**
   ```bash
   python chatbot.py
   ```

## Model Details

- **Base Model:** Mistral-7B-Instruct-v0.3 (free, open-source)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation) - efficient and memory-friendly
- **Training:** Uses 4-bit quantization to reduce memory requirements

## Requirements

- Python 3.8+
- **NVIDIA GPU with CUDA support (required)**
  - Minimum: 6GB+ VRAM (with 4-bit quantization)
  - Recommended: 8GB+ VRAM for smoother training
  - PyTorch with CUDA support installed
- Hugging Face account and token (for downloading models)
- ~10-20GB free disk space (for model and checkpoints)

### GPU Setup

This script requires a CUDA-capable GPU. To verify your setup:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If CUDA is not available:
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit
3. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace `cu118` with your CUDA version)

## Notes

- **Training time**: Depends on your dataset size and GPU. With a modern GPU, expect 1-3 hours for ~5000 examples.
- **Memory usage**: The 4-bit quantized model uses ~4-6GB VRAM. If you get out-of-memory errors:
  - Reduce `per_device_train_batch_size` in `train_zeller.py`
  - Increase `gradient_accumulation_steps` to maintain effective batch size
  - Close other applications using GPU memory
- **Model output**: The trained model will be saved in `./zeller-mistral` directory
- **Adjustments**: You can modify training parameters in `train_zeller.py`:
  - `num_train_epochs`: Number of training epochs (default: 3)
  - `learning_rate`: Learning rate (default: 2e-4)
  - `per_device_train_batch_size`: Batch size per GPU (default: 2)
  - `fp16`: Use mixed precision training (default: True)
  - `bf16`: Use bfloat16 if you have Ampere+ GPU (RTX 30xx, A100, etc.)

