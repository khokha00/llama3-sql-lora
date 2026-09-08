---
license: llama3
base_model: meta-llama/Meta-Llama-3-8B-Instruct
tags:
  - lora
  - qlora
  - peft
  - text-to-sql
  - sql
  - llama-3
datasets:
  - gretelai/synthetic_text_to_sql
language:
  - en
library_name: peft
pipeline_tag: text-generation
---

# Llama-3-8B-Instruct — Text-to-SQL LoRA Adapter

This is a QLoRA fine-tuned adapter for `meta-llama/Meta-Llama-3-8B-Instruct`,
trained to translate natural-language questions plus a database schema into
correct SQL queries.

## Model Details

- **Base model:** meta-llama/Meta-Llama-3-8B-Instruct
- **Fine-tuning method:** QLoRA (4-bit NF4 base weights, LoRA adapters on all
  attention + MLP projection layers)
- **LoRA rank / alpha:** r=16, alpha=32
- **Training data:** [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
  (subset of ~8,000 examples)
- **Hardware:** single Colab GPU (T4 / A100)
- **Adapter size:** ~[FILL IN — e.g. 42MB]

## Intended Use

Given a database schema and a natural-language question, generate an
executable SQL query. Intended for prototyping, developer tools, and as a
learning reference for QLoRA fine-tuning — **not** validated for production
use against untrusted or high-stakes databases.

### Prompt format

```
### Database Schema:
<CREATE TABLE statements>

### Question:
<natural language question>
```

The model was trained with Llama-3's chat template; use
`tokenizer.apply_chat_template` with a system message, this user content, and
generate the assistant turn.

## How to Use

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("<your-username>/llama3-8b-sql-lora")
tokenizer = AutoTokenizer.from_pretrained("<your-username>/llama3-8b-sql-lora")

messages = [
    {"role": "system", "content": "You are a helpful assistant that translates natural language questions into correct, executable SQL queries given a database schema."},
    {"role": "user", "content": "### Database Schema:\nCREATE TABLE employees (id INT, name TEXT, salary INT)\n\n### Question:\nWhat is the average salary?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Procedure

- Quantization: 4-bit NF4, double quantization, bf16 compute dtype
- Optimizer: paged AdamW 8-bit
- Epochs: 3
- Effective batch size: 16 (4 × 4 gradient accumulation)
- Learning rate: 2e-4, cosine schedule, 3% warmup

Full config: see `config.yaml` in the [training repo](<link to your GitHub repo>).

## Evaluation Results

*(Filled in after Week 8 — see `eval_harness.py`)*

| Metric | Base Model | Fine-tuned (this adapter) |
|---|---|---|
| Exact Match | [FILL IN] | [FILL IN] |
| Execution Accuracy | [FILL IN] | [FILL IN] |
| ROUGE-L | [FILL IN] | [FILL IN] |

## Limitations

- Trained on synthetic data; may not generalize to highly complex, real-world
  enterprise schemas.
- Does not validate that generated SQL is safe to execute — always run
  generated queries in a sandboxed / read-only environment.
- English-language questions only.

## Citation

If you use this adapter, please cite the base model (Meta Llama 3) and the
training dataset (Gretel AI synthetic text-to-SQL).