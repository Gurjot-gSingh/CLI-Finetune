import os
import sys
import json
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Same as training
ADAPTER_PATH = "project-root/training/lora-adapter"
LOGS_PATH = "logs"
LOG_FILE = os.path.join(LOGS_PATH, "trace.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_plan(tokenizer, model, instruction, max_length=256):
    prompt = (
        "You are an assistant that creates detailed step-by-step plans for command-line tasks.\n"
        "Instruction: " + instruction + "\n"
        "Complete the task by providing a clear numbered step-by-step plan:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the plan part after prompt
    plan = decoded[len(prompt):].strip()
    return plan

def is_shell_command(line):
    # Simple heuristic: line starts with common shell keywords or characters
    shell_starts = ['cd ', 'ls', 'echo', 'rm ', 'mkdir', 'git ', 'curl', 'tar ',
                    'gzip', './', 'python', 'chmod', 'mv ', 'cp ', 'sudo ', 'touch',
                    'cat ', 'head', 'tail', 'grep ', 'find ']
    return any(line.strip().startswith(prefix) for prefix in shell_starts)

def log_steps(steps):
    os.makedirs(LOGS_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for step in steps:
            json_line = json.dumps({"step": step})
            f.write(json_line + "\n")

def main():
    parser = argparse.ArgumentParser(description="CLI agent for command-line task planning with fine-tuned model.")
    parser.add_argument("instruction", type=str, help="Natural language instruction")
    args = parser.parse_args()

    tokenizer, model = load_model()
    plan_text = generate_plan(tokenizer, model, args.instruction)

    # Split plan into lines or steps by numbered points or newlines
    steps = []
    # Try splitting by numbers e.g. "1. ", "2. " etc
    split_by_nums = re.split(r'\n*\d+\.\s+', plan_text)
    if len(split_by_nums) > 1:
        steps = [step.strip() for step in split_by_nums if step.strip()]
    else:
        # Fallback to lines
        steps = [line.strip() for line in plan_text.split('\n') if line.strip()]

    # Log steps to logs/trace.jsonl
    log_steps(steps)

    # If first step looks like shell command, echo in dry-run
    if steps and is_shell_command(steps[0]):
        print(f"Dry-run shell command: echo {steps[0]}")
        for i, s in enumerate(steps[1:], start=2):
            print(f"Step {i}: {s}")
    else:
        # Print all steps normally
        for i, s in enumerate(steps, start=1):
            print(f"Step {i}: {s}")

if __name__ == "__main__":
    main()
