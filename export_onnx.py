"""
export_onnx.py — Week 9

1. Merges the LoRA adapter weights into the base model (LoRA adapters alone
   can't be exported to ONNX directly — they need to be fused first).
2. Exports the merged model to ONNX using Hugging Face Optimum.
3. Applies dynamic quantization (INT8 by default) to the ONNX graph.

Usage:
    python export_onnx.py --adapter_path ./output/final_adapter --quantize int8
    python export_onnx.py --adapter_path ./output/final_adapter --quantize none
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def merge_adapter(adapter_path: str, merged_dir: str):
    """Load base model in full precision, apply LoRA adapter, merge, and save."""
    print("Loading base model in fp16 for merging (needs ~16GB RAM/VRAM)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print("Applying LoRA adapter and merging weights...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged full-precision model saved to {merged_dir}")


def export_to_onnx(merged_dir: str, onnx_dir: str):
    print("Exporting merged model to ONNX (this can take a while for an 8B model)...")
    ort_model = ORTModelForCausalLM.from_pretrained(
        merged_dir, export=True, use_cache=True
    )
    ort_model.save_pretrained(onnx_dir)

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    tokenizer.save_pretrained(onnx_dir)
    print(f"ONNX model saved to {onnx_dir}")


def quantize_onnx(onnx_dir: str, quantized_dir: str, mode: str):
    print(f"Applying {mode.upper()} dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)

    if mode == "int8":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)
    print(f"Quantized ONNX model saved to {quantized_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="./output/final_adapter")
    parser.add_argument("--merged_dir", type=str, default="./merged_model")
    parser.add_argument("--onnx_dir", type=str, default="./onnx_export/fp32")
    parser.add_argument("--quantized_dir", type=str, default="./onnx_export/int8")
    parser.add_argument("--quantize", type=str, default="int8", choices=["int8", "none"])
    parser.add_argument("--skip_merge", action="store_true",
                         help="Skip merging if ./merged_model already exists")
    args = parser.parse_args()

    if not args.skip_merge:
        merge_adapter(args.adapter_path, args.merged_dir)

    export_to_onnx(args.merged_dir, args.onnx_dir)

    if args.quantize != "none":
        quantize_onnx(args.onnx_dir, args.quantized_dir, args.quantize)

    print("\nDone. ONNX exports:")
    print(f"  FP32: {args.onnx_dir}")
    if args.quantize != "none":
        print(f"  {args.quantize.upper()}: {args.quantized_dir}")


if __name__ == "__main__":
    main()