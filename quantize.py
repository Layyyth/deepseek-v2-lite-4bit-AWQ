from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Configuration
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
quant_path = "./DeepSeek-R1-Distill-Qwen-7B-AWQ"

# Load model and tokenizer
print(f"Loading model: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    safetensors=True
)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
print("Starting AWQ quantization...")
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
print(f"Saving to {quant_path}...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print("✅ AWQ Quantization Complete!")
print(f"Model saved to: {quant_path}")
