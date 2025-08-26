import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# -------------------------------
# Configuration
# -------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SAVE_DIR = "./DeepSeek-R1-Distill-Qwen-7B-AWQ"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"found device: {device}")

# Load model and tokenizer
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
# Calibration Data (Local, No HF Streaming)
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
# ✅ CORRECT WAY: Use Recipe String
# -------------------------------
recipe_str = """
- AWQModifier:
    ignore: ["lm_head"]
    num_samples: 32
    nsamples: 32
    seq_len: 512
    pad_to_max_length: true
"""

# Apply quantization
print("Applying AWQ via recipe...")
oneshot(
    model=model,
    recipe=recipe_str,
    calib_data=calib_texts,
    calib_func=calib_func,
    num_calib_steps=32,
)

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Quantized model saved to {SAVE_DIR}")
