# quantize.py (fixed for 429 error)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from llmcompressor
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot

# -------------------------------
# Configuration
# -------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "./DeepSeek-R1-Distill-Qwen-7B-AWQ"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Critical for calibration

# -------------------------------
# 🔥 Calibration Data: No Hugging Face Streaming!
# -------------------------------
print("Using local calibration prompts...")

# Tiny calibration dataset (no network needed)
calib_texts = [
    "The capital of France is Paris.",
    "Solve step by step: 123456789 ÷ 11. First, apply divisibility rule...",
    "Explain how a transformer model works in NLP.",
    "Write a Python function to compute Fibonacci numbers.",
    "What is the square root of 144? Let's think step by step.",
    "Describe the process of photosynthesis in plants.",
    "How does gravity affect planetary motion?",
    "Translate this to French: 'I am learning machine learning.'",
    "Explain the difference between supervised and unsupervised learning.",
    "Debug the following Python code: def add(a, b): return a - b"
] * 4  # 40 samples

def calib_func(model):
    model.eval()
    for text in calib_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(model.device)
        with torch.no_grad():
            model(**inputs)

# -------------------------------
# Apply AWQ
# -------------------------------
print("Applying AWQ...")
awq_modifier = AWQModifier(
    bits=4,
    group_size=128,
    zero_point=True,
    calib_data=calib_texts,
)

oneshot(
    model=model,
    recipe=awq_modifier,
    calib_func=calib_func,
    num_calib_steps=len(calib_texts),
    use_zephyr_chat_template=False,
)

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ AWQ Quantization Complete!")
