from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch

# 1. Load and preprocess 14 GB corpus (assuming a single text file or folder of files)
def load_large_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().splitlines()
    return {"text": text}

nepali_data = load_large_corpus("nepali_data.csv")  # Adjust for your file
dataset = Dataset.from_dict(nepali_data)
train_val_split = dataset.train_test_split(test_size=0.1)  # 12.6 GB train, 1.4 GB val
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

# 2. Load Mistral 7B with quantization
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
)

# Add LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Tokenize (batch processing for large data)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
tokenized_train.set_format("torch")
tokenized_val.set_format("torch")

# 4. Training setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./nepali_mistral_finetune",
    num_train_epochs=1,  # 14 GB is plenty for 1 epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=1000,
    save_steps=2000,
    warmup_steps=500,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# 5. Train
trainer.train()

# 6. Save
model.save_pretrained("./nepali_mistral_finetune/final_model")
tokenizer.save_pretrained("./nepali_mistral_finetune/final_model")

# # 7. Test
# from transformers import pipeline
# generator = pipeline("text-generation", model="./nepali_mistral_finetune/final_model", tokenizer=tokenizer)
# output = generator("नेपाल एक सुन्दर देश हो", max_length=100)
# print(output[0]["generated_text"])

print("Done")