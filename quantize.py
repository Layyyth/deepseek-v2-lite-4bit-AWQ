import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import AWQModifier
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

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Required for batch calibration

# -------------------------------
# Calibration Dataset
# -------------------------------
print("Loading calibration dataset (c4)...")
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib_data = [next(iter(dataset))["text"] for _ in range(512)]  # 512 samples

def calib_func(model):
    model.eval()
    for text in calib_data:
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
# AWQ via LLMCompressor
# -------------------------------
print("Starting AWQ calibration with llmcompressor...")
awq_modifier = AWQModifier(
    ignore=["lm_head"],           # Don't quantize head
    num_samples=512,              # Number of calibration samples
    nsamples=512,                 # Legacy alias
    seq_len=512,                  # Sequence length
    pad_to_max_length=True,
)

oneshot(
    model=model,
    recipe=awq_modifier,
    calib_data=calib_data,
    calib_func=calib_func,
    num_calib_steps=512,
)

# -------------------------------
# Save Quantized Model
# -------------------------------
print(f"Saving AWQ quantized model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ AWQ quantization complete using llmcompressor!")