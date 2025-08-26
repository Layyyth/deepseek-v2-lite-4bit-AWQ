from llm_compressor.recipes import AWQModifier, QuantizationModifier
from llm_compressor.engine import Compressor
from transformers import AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_path = "./llama3-8b-instruct-awq"

# Define the compression recipe, using AWQ for 4-bit weights
recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    QuantizationModifier(
        scheme="W4A16_ASYM",
        targets=[".*"], # Apply quantization to all layers except those ignored by AWQ
        group_size=128,
    ),
]

# Initialize the compressor
compressor = Compressor(
    recipe=recipe,
    model_name=model_name,
    quantized_model_path=quantized_model_path,
)

# Run the compression
compressor.run()

# Save the tokenizer and quantization config
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(quantized_model_path)
