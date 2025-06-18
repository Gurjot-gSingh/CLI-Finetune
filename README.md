#  CLI Agent for Command-Line Tasks (Fine-Tuned TinyLlama)

This project was built as part of the AI/ML Internship Technical Task for **Fenrir Security Pvt. Ltd.** It demonstrates the end-to-end process of fine-tuning a lightweight LLM (TinyLlama-1.1B) on command-line Q&A data, then wrapping it in an interactive terminal-based agent that performs dry-run shell simulations.

---

## 🚀 Project Overview

**Objective:**  
- Fine-tune a ≤2B open-source model on 150+ command-line Q&A pairs.
- Build a `CLI agent` that:
  - Accepts natural language input.
  - Generates a step-by-step plan.
  - Executes dry-run shell commands (`echo <cmd>`).
  - Logs all interactions in JSONL format.

---

## 🗂️ Project Structure
project-root/
├── agent.py # CLI agent interface
├── data/
│ └── qa_dataset.json # 150+ Q&A pairs (Git, Bash, grep, venv, etc.)
├── logs/
│ └── trace.jsonl # Session logs (input/output)
├── training/
│ ├── train_lora.py # LoRA fine-tuning script
│ └── lora-adapter/ # Fine-tuned adapter weights
├── eval_static.md # Base vs fine-tuned output comparison
├── eval_dynamic.md # Live agent evaluation results
├── report.md # Summary of process, hyperparams, time, cost, improvements
├── README.md # This file
├── requirements.txt # All dependencies


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/cli-agent-llm.git
cd cli-agent-llm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
🤖 Running the CLI Agent
```bash
python agent.py "Create a new Git branch and switch to it"
```

# Model Details
Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fine-Tuning: 1 epoch with LoRA using PEFT

Training: Done on Google Colab (T4 GPU)

Adapter Size: < 500MB
