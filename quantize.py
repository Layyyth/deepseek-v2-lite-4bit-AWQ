from llm_compressor import apply_awq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------------
# Configuration
# -------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "./DeepSeek-R1-Distill-Qwen-7B-AWQ-4bit"

# Load model and tokenizer
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Fix tokenizer (required)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------------------
# Calibration Data (Small & Local)
# -------------------------------
print("Preparing calibration data...")
calib_texts = [
    "The capital of France is Paris.",
    "Solve step by step: 123456789 ÷ 11.",
    "Explain how a transformer works.",
    "Write a Python function for Fibonacci.",
    "What is 144 squared? Think step by step.",
    "Describe photosynthesis.",
    "How does gravity work?",
    "Translate: 'I am learning AI.'",
    "Debug: def add(a, b): return a - b",
    "Explain supervised vs unsupervised learning."
] * 4  # 40 samples

def calib_func(model):
    model.eval()
    for text in calib_texts[:32]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            model(**inputs)

# -------------------------------
# Apply AWQ (Now Using vLLM's llm-compressor)
# -------------------------------
print("Applying AWQ quantization...")
apply_awq(
    model=model,
    tokenizer=tokenizer,
    data=calib_texts[:32],
    calib_iters=32,
    calib_batch_size=1,
    n_blocks=10,  # Use first 10 blocks for calibration (faster)
    w_bits=4,
    w_group_size=128,
    w_clip=False,
    w_zero_point=True,
    enable_quip=True,  # Use QUANTIZATION IF POSSIBLE
)

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ AWQ Quantization Complete! Model saved to {OUTPUT_DIR}")
