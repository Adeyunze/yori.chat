#!/usr/bin/env python3

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import argparse
from dotenv import load_dotenv

load_dotenv()

def load_training_data(data_path):
    """Load training data from JSONL file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_chat_template(example):
    """Format messages into a single text string."""
    messages = example['messages']
    text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            text += f"<|user|>\n{content}<|end|>\n"
        elif role == 'assistant':
            text += f"<|assistant|>\n{content}<|end|>\n"
    return {'text': text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--data_path', default='../data/yori_train.jsonl')
    parser.add_argument('--output_dir', default='./yori_adapter')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_length', type=int, default=512)
    args = parser.parse_args()

    # Load model and tokenizer with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    raw_data = load_training_data(args.data_path)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_chat_template)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True if torch.cuda.is_available() else False,
        warmup_steps=10,
        optim="paged_adamw_8bit"
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the adapter
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training completed. Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()