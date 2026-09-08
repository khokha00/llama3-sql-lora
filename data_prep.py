"""
data_prep.py — Week 7

Downloads gretelai/synthetic_text_to_sql, formats each example into a
Llama-3 chat-style prompt (system + user question + schema -> SQL answer),
and saves train/test splits to disk as Arrow datasets for training/eval.

Usage:
    python data_prep.py --config config.yaml
"""
import argparse
import yaml
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant that translates natural language questions "
    "into correct, executable SQL queries given a database schema."
)


def build_prompt(example):
    """Turn one raw dataset row into a Llama-3 chat-formatted training example."""
    schema = example.get("sql_context", "")
    question = example["sql_prompt"]
    sql_answer = example["sql"]

    user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sql_answer},
    ]
    return {"messages": messages}


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    print(f"Loading dataset: {ds_cfg['hf_dataset_id']}")
    raw = load_dataset(ds_cfg["hf_dataset_id"])

    train_ds = raw[ds_cfg["train_split"]]
    test_ds = raw[ds_cfg["test_split"]]

    if ds_cfg.get("max_train_samples"):
        train_ds = train_ds.shuffle(seed=42).select(
            range(min(ds_cfg["max_train_samples"], len(train_ds)))
        )
    if ds_cfg.get("max_eval_samples"):
        test_ds = test_ds.shuffle(seed=42).select(
            range(min(ds_cfg["max_eval_samples"], len(test_ds)))
        )

    print(f"Train examples: {len(train_ds)} | Test examples: {len(test_ds)}")

    train_ds = train_ds.map(build_prompt, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(build_prompt, remove_columns=test_ds.column_names)

    # Sanity check formatting with the target tokenizer's chat template
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_id"])
    sample_text = tokenizer.apply_chat_template(
        train_ds[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    print("\n--- Sample formatted example ---")
    print(sample_text[:800])
    print("--- end sample ---\n")

    out = DatasetDict({"train": train_ds, "test": test_ds})
    out.save_to_disk("./data/text_to_sql_formatted")
    print("Saved formatted dataset to ./data/text_to_sql_formatted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)