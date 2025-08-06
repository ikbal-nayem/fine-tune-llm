import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define your base model and fine-tuned adapter paths
base_model_path = "Qwen/Qwen3-0.6B"
lora_adapter_path = "./finetuned-model" # Path where your adapter_config.json and adapter_model.safetensors are saved
output_merged_model_path = "./qwen3-0.6b-merged" # Directory to save the merged model

print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, # Or torch.bfloat16, depending on your training
    device_map="auto" # Use "auto" to distribute across available GPUs or CPU
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print(f"Loading PEFT adapter from: {lora_adapter_path}")
model = PeftModel.from_pretrained(model, lora_adapter_path)

print("Merging LoRA weights...")
model = model.merge_and_unload() # This merges the LoRA weights into the base model

# Optional: Save in float32 for GGUF conversion, or keep float16 if your system handles it
# You might consider saving in float32 for maximum compatibility with llama.cpp conversion.
# model.to(torch.float32)

print(f"Saving merged model to: {output_merged_model_path}")
model.save_pretrained(output_merged_model_path)
tokenizer.save_pretrained(output_merged_model_path)

print("LoRA merging complete. Merged model saved.")