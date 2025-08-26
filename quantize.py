import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
tokenizer.padding_side = "left"  # Required for calibration

# -------------------------------
# Calibration Data (Local, No HF Streaming)
# -------------------------------
print("Using local calibration prompts...")
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
] * 4  # ~40 samples

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
# ✅ CORRECT WAY: Use Recipe String
# -------------------------------
print("Creating AWQ recipe...")
awq_recipe = """
quantization:
  ignore: ["lm_head"]
  config_groups:
    group_0:
      bits: 4
      group_size: 128
      scheme: sym
      format: affine
  calib_config:
    num_samples: 32
    forward_passes: 1

modifiers:
  - !AWQModifier
    name: 'AWQ'
    bits: 4
    group_size: 128
    zero_point: true
    calib_data: null  # Will be passed via oneshot
    model_fqn: null
    n_samples: 32
    seq_len: 512
"""

# -------------------------------
# Apply Quantization
# -------------------------------
from llmcompressor import oneshot

print("Applying AWQ via recipe...")
oneshot(
    model=model,
    recipe=awq_recipe,
    calib_func=calib_func,
    calib_data=calib_texts[:32],  # Pass actual data here
    num_calib_steps=32,
    use_zephyr_chat_template=False,
)

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving quantized model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ AWQ Quantization Complete!")
