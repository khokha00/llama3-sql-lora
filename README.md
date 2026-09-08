# Llama-3-8B Text-to-SQL — LoRA Fine-Tune, Eval, and ONNX Deployment

A 3-week build: fine-tune Llama-3-8B into a text-to-SQL model with QLoRA, rigorously
benchmark it against the base model, then export it to ONNX and measure the
inference speedup from quantization.

## Why text-to-SQL?
- Objective, automatic scoring (execution accuracy / exact match) — no need for
  human eval or an LLM judge.
- Small, clean public dataset (`gretelai/synthetic_text_to_sql`).
- A believable, demoable use case for a model card and a portfolio.

## Plan

| Week | Goal | Script(s) |
|------|------|-----------|
| 7 | QLoRA fine-tune Llama-3-8B, publish adapter + model card to Hugging Face | `data_prep.py`, `train_qlora.py`, `push_to_hub.py`, `MODEL_CARD.md` |
| 8 | Build an eval harness, report benchmark numbers vs. base model | `eval_harness.py` |
| 9 | Export to ONNX, quantize, benchmark inference speed | `export_onnx.py`, `benchmark_inference.py` |

## Requirements
See `requirements.txt`. Designed to run on **Google Colab** (a single T4 is enough
for QLoRA 4-bit training and inference on Llama-3-8B; an A100 — Colab Pro — will be
noticeably faster).

## Quickstart (Colab)

```bash
!git clone <your-repo-url> llama3-sql-lora
%cd llama3-sql-lora
!pip install -r requirements.txt

# Week 7
!python data_prep.py
!python train_qlora.py --config config.yaml
!python push_to_hub.py --repo_id <your-hf-username>/llama3-8b-sql-lora

# Week 8
!python eval_harness.py --model_path ./output/final_adapter --n_samples 200

# Week 9
!python export_onnx.py --adapter_path ./output/final_adapter --quantize int8
!python benchmark_inference.py --onnx_dir ./onnx_export
```

## Notes on Llama-3 access
`meta-llama/Meta-Llama-3-8B-Instruct` is a gated model on Hugging Face. Request
access on the model page, then run `huggingface-cli login` (or set the
`HF_TOKEN` env var / Colab secret) before downloading it.

## Repo structure

```
llama3-sql-lora/
├── README.md
├── requirements.txt
├── config.yaml               # all training hyperparameters
├── data_prep.py               # download + format the SQL dataset
├── train_qlora.py             # QLoRA fine-tuning (Wk7)
├── push_to_hub.py             # push adapter + card to HF Hub (Wk7)
├── MODEL_CARD.md              # HF model card template (Wk7)
├── eval_harness.py            # base vs fine-tuned benchmark (Wk8)
├── export_onnx.py             # ONNX export + quantization (Wk9)
└── benchmark_inference.py     # latency/throughput benchmark (Wk9)
```