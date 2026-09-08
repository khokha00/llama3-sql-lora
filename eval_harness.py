"""
eval_harness.py — Week 8

Benchmarks the fine-tuned adapter against the base Llama-3-8B-Instruct model
on held-out text-to-SQL examples. Reports:
  - Exact Match (normalized SQL string match)
  - Execution Accuracy (results match when run against a scratch SQLite DB
    built from the example's schema, when possible)
  - ROUGE-L (surface-level similarity, as a softer secondary signal)

Usage:
    python eval_harness.py --model_path ./output/final_adapter --n_samples 200
    python eval_harness.py --base_only --n_samples 200
"""
import argparse
import json
import re
import sqlite3
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate
import sqlglot

SYSTEM_PROMPT = (
    "You are a helpful assistant that translates natural language questions "
    "into correct, executable SQL queries given a database schema."
)
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_model(model_path: str, is_adapter: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    if is_adapter:
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = base
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if is_adapter else BASE_MODEL_ID
    )
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, schema: str, question: str, max_new_tokens=200) -> str:
    user_content = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite")[0].strip().lower()
    except Exception:
        return re.sub(r"\s+", " ", sql).strip().lower().rstrip(";")


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def try_execution_match(pred: str, gold: str, schema: str) -> bool | None:
    """Run both queries against a scratch in-memory SQLite DB built from `schema`.
    Returns True/False if executable, None if the schema/query can't run in SQLite
    (e.g. dialect-specific syntax) — those examples are excluded from the metric.
    """
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.executescript(schema)
        conn.commit()

        pred_rows = cur.execute(pred).fetchall()
        gold_rows = cur.execute(gold).fetchall()
        conn.close()
        return sorted(map(str, pred_rows)) == sorted(map(str, gold_rows))
    except Exception:
        return None


def run_eval(model, tokenizer, dataset, n_samples: int):
    rouge = evaluate.load("rouge")
    preds, golds = [], []
    exact_matches = 0
    exec_matches, exec_total = 0, 0

    n = min(n_samples, len(dataset))
    for i in range(n):
        example = dataset[i]
        # Recover schema/question/gold from the stored chat messages
        messages = example["messages"]
        user_msg = messages[1]["content"]
        gold_sql = messages[2]["content"]
        schema = user_msg.split("### Question:")[0].replace("### Database Schema:\n", "").strip()
        question = user_msg.split("### Question:\n")[-1].strip()

        pred_sql = generate_sql(model, tokenizer, schema, question)
        preds.append(pred_sql)
        golds.append(gold_sql)

        if exact_match(pred_sql, gold_sql):
            exact_matches += 1

        exec_result = try_execution_match(pred_sql, gold_sql, schema)
        if exec_result is not None:
            exec_total += 1
            if exec_result:
                exec_matches += 1

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{n} evaluated")

    rouge_scores = rouge.compute(predictions=preds, references=golds)

    return {
        "n_samples": n,
        "exact_match": round(exact_matches / n, 4),
        "execution_accuracy": round(exec_matches / exec_total, 4) if exec_total else None,
        "execution_eval_coverage": round(exec_total / n, 4),
        "rougeL": round(rouge_scores["rougeL"], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output/final_adapter")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--base_only", action="store_true",
                         help="Evaluate only the base model, skip the adapter")
    parser.add_argument("--output_json", type=str, default="./eval_results.json")
    args = parser.parse_args()

    dataset = load_from_disk("./data/text_to_sql_formatted")["test"]

    results = {}

    print("Evaluating BASE model...")
    base_model, base_tok = load_model(BASE_MODEL_ID, is_adapter=False)
    results["base_model"] = run_eval(base_model, base_tok, dataset, args.n_samples)
    del base_model
    torch.cuda.empty_cache()

    if not args.base_only:
        print("Evaluating FINE-TUNED (LoRA) model...")
        ft_model, ft_tok = load_model(args.model_path, is_adapter=True)
        results["fine_tuned_model"] = run_eval(ft_model, ft_tok, dataset, args.n_samples)

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()