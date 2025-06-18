import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DATA_FILE = "project-root\data\data.json"
OUTPUT_DIR = "project-root\training"
EPOCHS = 1
BATCH_SIZE = 4  # Adjust per your VRAM
MAX_LEN = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = [{"text": f"Q: {item['question']}\nA: {item['answer']}"} for item in data]

    dataset = Dataset.from_list(formatted_data)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['input_ids'] for f in data])
        },
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
