from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import HfFolder, Repository
import shutil
import os

base_model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
quantized_dir = "./DeepSeek-V2-Lite-Chat-4bit-vllm"
upload_repo_id = "LaythAbuJafar/Deepseek-v2-lite-4bit-AWQ"

# Load tokenizer for saving later
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Initialize vLLM LLM, specifying quantization (AWQ or GPTQ)
llm = LLM(
    model=base_model,
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True,
    quantize='awq'  # or 'gptq' if you want that
)

# Run a dummy generation (optional) to initialize weights in vLLM
prompt = "Hello world!"
sampling_params = SamplingParams(temperature=0.3, max_tokens=32)
outputs = llm.generate([prompt], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)

# Save quantized weights
llm.save_quantized(quantized_dir)
tokenizer.save_pretrained(quantized_dir)
print(f"Quantized model and tokenizer saved to {quantized_dir}")

# Prepare pushing to HF Hub

# Clone repo or create if not exists locally
if os.path.exists(quantized_dir + "/.git"):
    print("Using existing git repo")
else:
    repo = Repository(quantized_dir, clone_from=upload_repo_id)
    print(f"Cloned repo {upload_repo_id}")

# Login with huggingface CLI beforehand: `huggingface-cli login`
# Push your model
repo = Repository(quantized_dir)
repo.push_to_hub(commit_message="Add 4-bit AWQ quantized DeepSeek-V2-Lite-Chat model")

print(f"Model pushed to https://huggingface.co/{upload_repo_id}")
