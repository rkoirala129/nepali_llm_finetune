import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model path (use your fine-tuned model or the original)
model_path = "meta-llama/Llama-3.2-1B"  # Change to "mistralai/Mistral-7B-v0.1" if not fine-tuned yet

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token="hf_token")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token exists

# Load the model with optimizations for RTX 4090
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,         # 4-bit quantization to fit in 24 GB VRAM
    torch_dtype=torch.float16, # Mixed precision for speed
    device_map="auto",         # Auto-map to CUDA (RTX 4090)
)

# Ensure model is on GPU
model = model.to("cuda")

# Generate text
prompt = "नेपाल एक सुन्दर देश हो"  # "Nepal is a beautiful country"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,    # Enable sampling for diverse outputs
    top_k=50,          # Top-k sampling
    top_p=0.95,        # Top-p (nucleus) sampling
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
