"""
benchmark_inference.py — Week 9

Benchmarks inference latency and throughput across:
  1. Original merged PyTorch model (fp16)
  2. ONNX export (fp32)
  3. ONNX export (int8 quantized)

Reports mean/p50/p95 latency per generation and tokens/sec throughput.

Usage:
    python benchmark_inference.py --merged_dir ./merged_model \
                                   --onnx_fp32_dir ./onnx_export/fp32 \
                                   --onnx_int8_dir ./onnx_export/int8 \
                                   --n_runs 20
"""
import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

BENCHMARK_PROMPTS = [
    "### Database Schema:\nCREATE TABLE employees (id INT, name TEXT, salary INT, dept TEXT)\n\n### Question:\nWhat is the average salary per department?",
    "### Database Schema:\nCREATE TABLE orders (id INT, customer_id INT, total REAL, created_at DATE)\n\n### Question:\nHow many orders were placed in the last 30 days?",
    "### Database Schema:\nCREATE TABLE products (id INT, name TEXT, price REAL, stock INT)\n\n### Question:\nList the 5 most expensive products currently in stock.",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant that translates natural language questions "
    "into correct, executable SQL queries given a database schema."
)


def build_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def benchmark_model(model, tokenizer, n_runs: int, max_new_tokens: int, device: str):
    latencies = []
    total_tokens = 0

    # Warmup
    warmup_prompt = build_prompt(tokenizer, BENCHMARK_PROMPTS[0])
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)

    for i in range(n_runs):
        prompt = build_prompt(tokenizer, BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        total_tokens += output.shape[1] - inputs["input_ids"].shape[1]

    latencies = np.array(latencies)
    return {
        "mean_latency_s": round(float(latencies.mean()), 4),
        "p50_latency_s": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 4),
        "tokens_per_sec": round(total_tokens / latencies.sum(), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", type=str, default="./merged_model")
    parser.add_argument("--onnx_fp32_dir", type=str, default="./onnx_export/fp32")
    parser.add_argument("--onnx_int8_dir", type=str, default="./onnx_export/int8")
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--skip_pytorch", action="store_true",
                         help="Skip the fp16 PyTorch baseline (slow to load on top of ONNX runs)")
    parser.add_argument("--output_json", type=str, default="./inference_benchmark.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    if not args.skip_pytorch:
        print("Benchmarking PyTorch (fp16) model...")
        tokenizer = AutoTokenizer.from_pretrained(args.merged_dir)
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.merged_dir, torch_dtype=torch.float16
        ).to(device)
        pt_model.eval()
        results["pytorch_fp16"] = benchmark_model(
            pt_model, tokenizer, args.n_runs, args.max_new_tokens, device
        )
        del pt_model
        torch.cuda.empty_cache()

    print("Benchmarking ONNX (fp32) model...")
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_fp32_dir)
    onnx_fp32_model = ORTModelForCausalLM.from_pretrained(args.onnx_fp32_dir)
    results["onnx_fp32"] = benchmark_model(
        onnx_fp32_model, tokenizer, args.n_runs, args.max_new_tokens, "cpu"
    )
    del onnx_fp32_model

    print("Benchmarking ONNX (int8 quantized) model...")
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_int8_dir)
    onnx_int8_model = ORTModelForCausalLM.from_pretrained(args.onnx_int8_dir)
    results["onnx_int8"] = benchmark_model(
        onnx_int8_model, tokenizer, args.n_runs, args.max_new_tokens, "cpu"
    )

    print("\n=== Inference Benchmark Results ===")
    print(json.dumps(results, indent=2))

    if "pytorch_fp16" in results and "onnx_int8" in results:
        speedup = results["pytorch_fp16"]["mean_latency_s"] / results["onnx_int8"]["mean_latency_s"]
        print(f"\nONNX INT8 vs PyTorch FP16 speedup: {speedup:.2f}x")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()