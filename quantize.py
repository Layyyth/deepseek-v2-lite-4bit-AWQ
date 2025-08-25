# quantize.py
from llmcompressor import oneshot
from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
save_path = "Deepseek-v2-lite-4bit-AWQ"

# Load tokenizer with trust_remote_code
print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

# Define AWQ recipe (minimal and valid)
recipe_str = """
- QuantizationModifier:
    config_groups:
      group_0:
        weights:
          num_bits: 4
          type: uint
          symmetric: true
          strategy: tensor
          group_size: 128
          zero_point: true
    scheme:
      weights:
        type: llm.int4
    targets: ["Linear"]
    ignore:
      - model.encoder.final_layer_norm
      - model.encoder.layers.*.layer_norm
      - model.decoder.final_layer_norm
      - model.decoder.layers.*.layer_norm
    quantize_embeddings: false
    quantize_layer_norms: false
    quantize_output_logits: false
"""

print("🔥 Starting AWQ quantization...")
oneshot(
    model=model_name,
    dataset="c4",
    tokenizer=tokenizer,
    max_seq_length=512,
    num_calibration_samples=32,
    output_dir=save_path,
    recipe=recipe_str,
    # DO NOT pass: device, trust_remote_code (already in tokenizer)
)

print(f"✅ Quantization complete! Model saved to {save_path}")