from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


quantized_model = LLM(
    model=model_name,
    quantization="awq",  
    dtype="float16",     
    gpu_memory_utilization=0.9  
)


prompt = "Explain the concept of AI in simple terms."
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
outputs = quantized_model.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)


quantized_model.save("quantized-deepseek-v2-lite-chat")