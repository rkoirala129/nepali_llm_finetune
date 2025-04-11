from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both from the same directory
model = AutoModelForCausalLM.from_pretrained("./nepali_mistral_finetune/final_model")
tokenizer = AutoTokenizer.from_pretrained("./nepali_mistral_finetune/final_model")

# Test
prompt = "नेपाल एक सुन्दर देश हो"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))