import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Qwen/Qwen3-0.6B"
DATASET = "output-data/dataset-text.jsonl"
OUTPUT_DIR = "./finetuned-model"     # Where to save the fine-tuned model
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("json", data_files=DATASET, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir="./temp-cache"
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-4,
	bf16=USE_BF16,
	fp16=not USE_BF16 and DEVICE == "cuda",  # Fallback to FP16 if BF16 unavailable
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",                      # Disable logging to Hugging Face Hub
    push_to_hub=False
)

print("Start training...", flush=True)
model.print_trainable_parameters()

# SFTTrainer is a high-level trainer for instruction fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=lambda x: x["text"],
)
print("Done")

trainer.train()
trainer.save_model(OUTPUT_DIR)