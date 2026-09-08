"""
train_qlora.py — Week 7

QLoRA fine-tuning of Llama-3-8B-Instruct for text-to-SQL.
Loads the base model in 4-bit (NF4), attaches LoRA adapters, and trains
with TRL's SFTTrainer on the dataset produced by data_prep.py.

Usage:
    python train_qlora.py --config config.yaml
"""
import argparse
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_id"], trust_remote_code=model_cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def formatting_func(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    train_cfg = cfg["training"]

    print("Loading model + tokenizer in 4-bit (QLoRA)...")
    model, tokenizer = build_model_and_tokenizer(cfg)
    peft_config = build_lora_config(cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading formatted dataset (run data_prep.py first if missing)...")
    dataset = load_from_disk("./data/text_to_sql_formatted")

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        max_seq_length=cfg["dataset"]["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    final_dir = f"{train_cfg['output_dir']}/final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Adapter saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)