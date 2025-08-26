import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"found device: {device}")

# Select model and load it.
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Create the modifier
modifier = AWQModifier(
    ignore=["lm_head"],
    num_samples=32,
    nsamples=32,
    seq_len=512,
    pad_to_max_length=True,
)

# Dummy calibration data (avoid c4 rate limit)
calib_texts = [
    "The capital of France is Paris.",
    "Solve step by step: 123456789 ÷ 11.",
    "Explain how a transformer works."
] * 10  # 30 samples

def calib_func(model):
    model.eval()
    for text in calib_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            model(**inputs)

# Apply one-shot quantization
oneshot(
    model=model,
    recipe=modifier,
    calib_data=calib_texts,
    calib_func=calib_func,
    num_calib_steps=32,
)

# Save the quantized model
SAVE_DIR = "./DeepSeek-R1-Distill-Qwen-7B-AWQ"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Model saved to {SAVE_DIR}")
