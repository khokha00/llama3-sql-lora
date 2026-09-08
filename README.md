# Llama-3-8B Text-to-SQL

QLoRA fine-tune of Llama-3-8B-Instruct for translating natural language questions into SQL, with a benchmark eval harness and ONNX/INT8 export for fast inference.

## Features

- 4-bit QLoRA fine-tuning on [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Eval harness scoring exact match, SQL execution accuracy, and ROUGE-L against the base model
- ONNX export with INT8 dynamic quantization
- Inference latency/throughput benchmarking (PyTorch vs. ONNX FP32 vs. ONNX INT8)

## Repo structure

```
llama3-sql-lora/
├── requirements.txt
├── config.yaml               # all training hyperparameters
├── data_prep.py               # download + format the SQL dataset
├── train_qlora.py             # QLoRA fine-tuning
├── push_to_hub.py             # push adapter + card to HF Hub
├── MODEL_CARD.md              # HF model card
├── eval_harness.py            # base vs fine-tuned benchmark
├── export_onnx.py             # ONNX export + quantization
├── benchmark_inference.py     # latency/throughput benchmark
├── run_pipeline.sh            # chains every stage
└── Colab_Pipeline.ipynb       # same pipeline as a Colab notebook
```

## Setup

```bash
git clone <your-repo-url> llama3-sql-lora
cd llama3-sql-lora
pip install -r requirements.txt
huggingface-cli login
```

`meta-llama/Meta-Llama-3-8B-Instruct` is gated — request access on its
[model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
before running any script that downloads it.

## Usage

Run everything:

```bash
bash run_pipeline.sh --hf-repo <your-username>/llama3-8b-sql-lora
```

Or stage by stage:

```bash
python data_prep.py --config config.yaml
python train_qlora.py --config config.yaml
python push_to_hub.py --repo_id <your-username>/llama3-8b-sql-lora
python eval_harness.py --model_path ./output/final_adapter --n_samples 200
python export_onnx.py --adapter_path ./output/final_adapter --quantize int8
python benchmark_inference.py --onnx_fp32_dir ./onnx_export/fp32 --onnx_int8_dir ./onnx_export/int8
```

On Colab, open `Colab_Pipeline.ipynb`, set the runtime to GPU, add an
`HF_TOKEN` secret, and run the cells top to bottom.

## Results

See `eval_results.json` and `inference_benchmark.json` after running, or the
published model card at `https://huggingface.co/<your-username>/llama3-8b-sql-lora`.

## License

Base model is subject to the [Llama 3 license](https://llama.meta.com/llama3/license/).
