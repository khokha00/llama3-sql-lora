# Llama-3-8B Text-to-SQL — LoRA Fine-Tune, Eval, and ONNX Deployment

A 3-week build: fine-tune Llama-3-8B into a text-to-SQL model with QLoRA, rigorously
benchmark it against the base model, then export it to ONNX and measure the
inference speedup from quantization.

## Why text-to-SQL?
- Objective, automatic scoring (execution accuracy / exact match) — no need for
  human eval or an LLM judge.
- Small, clean public dataset (`gretelai/synthetic_text_to_sql`).
- A believable, demoable use case for a model card and a portfolio.

## Features

- 4-bit QLoRA fine-tuning on [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Eval harness scoring exact match, SQL execution accuracy, and ROUGE-L against the base model
- ONNX export with INT8 dynamic quantization
- Inference latency/throughput benchmarking (PyTorch vs. ONNX FP32 vs. ONNX INT8)

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

## Setup

```bash
git clone <your-repo-url> llama3-sql-lora
cd llama3-sql-lora
pip install -r requirements.txt
huggingface-cli login
```

`meta-llama/Meta-Llama-3-8B-Instruct` is gated — request access on its
[model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
before running any script that downloads it. You can also set the `HF_TOKEN`
env var or a Colab secret.

## Usage

```bash
python data_prep.py --config config.yaml
python train_qlora.py --config config.yaml
python push_to_hub.py --repo_id <your-username>/llama3-8b-sql-lora
python eval_harness.py --model_path ./output/final_adapter --n_samples 200
python export_onnx.py --adapter_path ./output/final_adapter --quantize int8
python benchmark_inference.py --onnx_fp32_dir ./onnx_export/fp32 --onnx_int8_dir ./onnx_export/int8
```

On Colab, clone the repo, set the runtime to GPU, add an `HF_TOKEN` secret, and
run the same commands with `!` prefixes.

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

## Results

See `eval_results.json` and `inference_benchmark.json` after running, or the
published model card at `https://huggingface.co/<your-username>/llama3-8b-sql-lora`.

## License

Base model is subject to the [Llama 3 license](https://llama.meta.com/llama3/license/).
