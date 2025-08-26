import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Define the model to be quantized
model_name = "Qwen/Qwen2-7B-Instruct"

# Define the output path for the quantized model
quantized_model_path = "./Qwen2-7B-Instruct-awq"

# Define the AWQ quantization recipe
# W4A16_ASYM means 4-bit weights, 16-bit activations, asymmetric quantization
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
        group_size=128,
    ),
]

# Run the one-shot quantization
# The library automatically downloads a calibration dataset from Hugging Face for this step
oneshot(
    model=model_name,
    dataset="open_platypus",  # A small calibration dataset
    recipe=recipe,
    output_dir=quantized_model_path,
    num_calibration_samples=512,
    max_seq_length=2048,
)

print(f"Quantized model saved to: {quantized_model_path}")
