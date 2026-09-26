#!/usr/bin/env python3
"""QLoRA fine-tuning used in the adaptation experiment.

Usage:
  python code/finetuning/lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct \
      --data data/finetuning/debias_sft.jsonl \
      --out_dir outputs/qwen2.5-7b-debias-lora
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                           DataCollatorForLanguageModeling, Trainer, TrainingArguments)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config,
                                                  device_map="auto", torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw = load_jsonl(args.data)

    def to_text(ex):
        messages = [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["completion"]}]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(raw).map(to_text)

    def tokenize(ex):
        # Use dynamic batch padding rather than padding every example to max_len.
        out = tokenizer(ex["text"], truncation=True, max_length=args.max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, learning_rate=args.lr, logging_steps=5, save_strategy="epoch",
        bf16=True, report_to=[], optim="paged_adamw_8bit", gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds,
                       data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
