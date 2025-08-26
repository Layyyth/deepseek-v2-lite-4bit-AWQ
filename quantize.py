import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
quant_path = "./DeepSeek-R1-Distill-Qwen-32B-AWQ"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  
}


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.half,
    trust_remote_code=True
)


print("Starting AWQ quantization...")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="wikitext2"  
)


model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"AWQ quantized model saved to {quant_path}")