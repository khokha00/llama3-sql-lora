"""
push_to_hub.py — Week 7

Pushes the trained LoRA adapter (and tokenizer) to the Hugging Face Hub,
and uploads MODEL_CARD.md as the repo's README.

Requires: huggingface-cli login (or HF_TOKEN env var / Colab secret) beforehand.

Usage:
    python push_to_hub.py --repo_id your-username/llama3-8b-sql-lora \
                           --adapter_path ./output/final_adapter \
                           --private false
"""
import argparse
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True,
                         help="e.g. your-username/llama3-8b-sql-lora")
    parser.add_argument("--adapter_path", type=str, default="./output/final_adapter")
    parser.add_argument("--model_card_path", type=str, default="MODEL_CARD.md")
    parser.add_argument("--private", type=str, default="false")
    args = parser.parse_args()

    private = args.private.lower() == "true"

    print(f"Creating (or reusing) repo: {args.repo_id} (private={private})")
    create_repo(args.repo_id, private=private, exist_ok=True)

    print("Loading adapter + tokenizer to verify they load cleanly before pushing...")
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    print("Pushing adapter weights...")
    model.push_to_hub(args.repo_id, private=private)
    tokenizer.push_to_hub(args.repo_id, private=private)

    print("Uploading model card as README.md...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=args.model_card_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Done. View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()