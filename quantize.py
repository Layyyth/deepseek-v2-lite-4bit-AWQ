import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variable to avoid HfArgumentParser issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model and tokenizer
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SAVE_DIR = "./DeepSeek-R1-Distill-Qwen-7B-AWQ"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Fix tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------------------
# Calibration Data
# -------------------------------
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
# ✅ Use Recipe String Only
# -------------------------------
recipe_str = """
- AWQModifier:
    ignore: ["lm_head"]
    num_samples: 32
    seq_len: 512
"""

# -------------------------------
# ✅ Manual AWQ Application
# -------------------------------
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.utils import apply_modifiers

# Parse recipe
modifier = AWQModifier.from_config({"modifiers": {"- AWQModifier": recipe_str}})

# Apply AWQ
print("Applying AWQ...")
modifier.model = model
modifier.tokenizer = tokenizer
modifier.calib_func = calib_func
modifier.apply()

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Quantized model saved to {SAVE_DIR}")
