#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Yori AI Companion.
Defaults to teknium/OpenHermes-2.5-Mistral-7B and can target any compatible causal LM.
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl_data(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load training data from JSONL file."""
    data = []
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and len(data) >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                if 'messages' in item and isinstance(item['messages'], list):
                    data.append(item)
                else:
                    logger.warning(f"Invalid format at line {line_num + 1}")
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error at line {line_num + 1}")
    
    logger.info(f"Loaded {len(data)} training examples")
    return data

def format_chat_template(example: Dict[str, Any]) -> Dict[str, str]:
    """Format messages into a single text string using Phi-3 chat template."""
    messages = example['messages']
    formatted_text = ""
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == 'system':
            formatted_text += f"<|system|>\n{content}<|end|>\n"
        elif role == 'user':
            formatted_text += f"<|user|>\n{content}<|end|>\n"
        elif role == 'assistant':
            formatted_text += f"<|assistant|>\n{content}<|end|>\n"
    
    return {'text': formatted_text}

def setup_model_and_tokenizer(model_name: str):
    """Set up the model and tokenizer with 4-bit quantization."""
    logger.info(f"Loading model: {model_name}")
    
    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Set up LoRA configuration for QLoRA."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )
    return lora_config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune teknium/OpenHermes-2.5-Mistral-7B with QLoRA")
    parser.add_argument("--data_path", type=str, default="./data/yori_train.jsonl", 
                       help="Path to training data JSONL file")
    parser.add_argument("--out_dir", type=str, default="./yori_flirty_adapter",
                       help="Output directory for saving adapter")
    parser.add_argument("--cut_len", type=int, default=2048,
                       help="Maximum sequence length for training")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of training samples to use")
    parser.add_argument("--model_name", type=str, default="teknium/OpenHermes-2.5-Mistral-7B",
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10,
                       help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="Save checkpoint frequency")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current device: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available, using CPU (training will be very slow)")
    
    # Load and prepare data
    raw_data = load_jsonl_data(args.data_path, args.max_samples)
    
    # Convert to Hugging Face dataset and format
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
    
    logger.info(f"Dataset prepared with {len(dataset)} examples")
    logger.info(f"Example formatted text: {dataset[0]['text'][:200]}...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        tf32=torch.cuda.is_available(),
        max_grad_norm=0.3,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        disable_tqdm=False,
        report_to=None,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Data collator for completion-only training
    response_template = "<|assistant|>"
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, 
        response_template=response_template
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=args.cut_len,
        dataset_text_field="text",
        packing=False,  # Set to True if you want to pack sequences
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Print training results
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training steps: {train_result.global_step}")
    
    # Save the final model
    logger.info(f"Saving adapter to: {args.out_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.out_dir)
    
    # Save training arguments for reference
    with open(os.path.join(args.out_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"✅ Training complete! Adapter saved to: {os.path.abspath(args.out_dir)}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Test the model with a simple example
    logger.info("Testing trained model...")
    try:
        test_prompt = "<|system|>\nYou are Yori, a warm, playful AI companion.<|end|>\n<|user|>\nHello!<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
    except Exception as e:
        logger.warning(f"Test generation failed: {e}")
    
    logger.info("🎉 All done! Your Yori model is ready to chat!")

if __name__ == "__main__":
    main()
