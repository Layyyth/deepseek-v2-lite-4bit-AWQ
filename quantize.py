from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

# Define model paths
base_model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
quantized_model_path = "DeepSeek-V2-Lite-Chat-4bit-AWQ"

# Define quantization configuration
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}

if __name__ == "__main__":
    print("=================================================================")
    print("          STARTING AWQ QUANTIZATION SCRIPT                       ")
    print("=================================================================")
    print(f"-> Base Model: {base_model_path}")
    print(f"-> Output Path: {quantized_model_path}")
    print(f"-> PyTorch Version: {torch.__version__}")
    print("-----------------------------------------------------------------")

    # Step 1: Load the base model and tokenizer
    print("\n[STEP 1/3] Loading base model and tokenizer...")
    try:
        model = AutoAWQForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("-> SUCCESS: Model and tokenizer loaded.")
    except Exception as e:
        print(f"-> ERROR: Failed to load model. {e}")
        exit()

    # Step 2: Quantize the model
    print("\n[STEP 2/3] Quantizing the model...")
    try:
        model.quantize(tokenizer, quant_config=quant_config)
        print("-> SUCCESS: Quantization complete.")
    except Exception as e:
        print(f"-> ERROR: Failed during quantization. {e}")
        exit()

    # Step 3: Save the quantized model and tokenizer
    print(f"\n[STEP 3/3] Saving quantized model to '{quantized_model_path}'...")
    try:
        model.save_quantized(quantized_model_path)
        tokenizer.save_pretrained(quantized_model_path)
        print(f"-> SUCCESS: Quantized model saved at '{quantized_model_path}'")
    except Exception as e:
        print(f"-> ERROR: Failed to save quantized model. {e}")
        exit()
