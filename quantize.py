# quantize.py
from llmcompressor import oneshot
from transformers import AutoTokenizer

# Model and output paths
model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
save_path = "Deepseek-v2-lite-4bit-AWQ"

# Load tokenizer
print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Define AWQ recipe as a YAML string
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
    ignore:
      - model.encoder.final_layer_norm
      - model.encoder.layers.*.layer_norm
      - model.decoder.final_layer_norm
      - model.decoder.layers.*.layer_norm
    scheme:
      weights:
        type: llm.int4
    targets: ["Linear"]
    quantize_embeddings: false
    quantize_layer_norms: false
    quantize_output_logits: false
    device: "cuda:0"
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
    device="cuda:0",
    trust_remote_code=True,
)

print(f"✅ Quantization complete! Model saved to {save_path}")