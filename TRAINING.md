# Training Guide for Yori.chat

This guide explains how to train and deploy custom LoRA adapters for the Yori AI companion.

## Quick Training Commands

### RunPod Setup
```bash
# 1. Activate venv
source venv/bin/activate

# 2. Train LoRA adapter
python training/train_qlora.py --data_path ./data/yori_train.jsonl --out_dir ./yori_flirty_adapter

# 3. Update .env
echo "BASE_MODEL=microsoft/Phi-3-mini-4k-instruct" >> .env
echo "ADAPTER_DIR=./yori_flirty_adapter" >> .env

# 4. Restart server and test
pkill -f uvicorn
cd server && uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 5
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "Hello Yori!", "user_id": "test_user"}'
```

## Training Script Overview

The `training/train_qlora.py` script provides QLoRA fine-tuning for microsoft/Phi-3-mini-4k-instruct:

### Key Features
- **QLoRA (4-bit)**: Memory-efficient training with bitsandbytes
- **Target Modules**: All linear layers (q,k,v,o,gate,up,down projections)
- **Chat Format**: Uses Phi-3 chat template (`<|system|>`, `<|user|>`, `<|assistant|>`)
- **Data Format**: JSONL with `{"messages": [...]}` structure
- **Completion Training**: Only trains on assistant responses

### Command Line Arguments
```bash
python training/train_qlora.py [OPTIONS]

Options:
  --data_path TEXT           Training data JSONL file [default: ./data/yori_train.jsonl]
  --out_dir TEXT            Output directory for adapter [default: ./yori_flirty_adapter]
  --cut_len INTEGER         Max sequence length [default: 2048]
  --max_samples INTEGER     Limit training samples [default: None]
  --model_name TEXT         Base model [default: microsoft/Phi-3-mini-4k-instruct]
  --batch_size INTEGER      Training batch size [default: 2]
  --gradient_accumulation_steps INTEGER  [default: 4]
  --num_train_epochs INTEGER  [default: 3]
  --learning_rate FLOAT     [default: 2e-4]
  --warmup_steps INTEGER    [default: 10]
  --logging_steps INTEGER   [default: 5]
  --save_steps INTEGER      [default: 50]
```

## Training Examples

### Basic Training
```bash
# Train with default settings
python training/train_qlora.py

# Train with custom data
python training/train_qlora.py \
  --data_path ./data/yori_train_companion.jsonl \
  --out_dir ./yori_companion_adapter \
  --num_train_epochs 2
```

### Advanced Training
```bash
# High-quality training with more epochs
python training/train_qlora.py \
  --data_path ./data/yori_train_extended.jsonl \
  --out_dir ./yori_production_adapter \
  --num_train_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4

# Quick test training with limited data
python training/train_qlora.py \
  --max_samples 50 \
  --num_train_epochs 1 \
  --out_dir ./yori_test_adapter
```

## Data Format

Training data must be in JSONL format with messages structure:

```json
{"messages": [
  {"role": "system", "content": "You are Yori, a warm, playful AI companion..."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there! How are you doing today? 😊"}
]}
```

### Available Training Datasets
- `data/yori_train.jsonl` - Original 20 examples
- `data/yori_train_extended.jsonl` - 100 diverse examples
- `data/yori_train_companion.jsonl` - 50 playful/flirty examples

## GPU Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, RTX 2080 Ti)
- **Settings**: `--batch_size 1 --gradient_accumulation_steps 8`
- **Training Time**: ~30-45 minutes for 100 steps

### Recommended Setup
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A100)
- **Settings**: `--batch_size 2-4 --gradient_accumulation_steps 4`
- **Training Time**: ~15-25 minutes for 100 steps

### High-End Setup
- **GPU**: 24GB+ VRAM (RTX 4090, A6000, H100)
- **Settings**: `--batch_size 4-8 --gradient_accumulation_steps 2`
- **Training Time**: ~10-15 minutes for 100 steps

## Memory Optimization Tips

### For Low VRAM (8GB)
```bash
python training/train_qlora.py \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --cut_len 1024 \
  --max_samples 100
```

### For Medium VRAM (12-16GB)
```bash
python training/train_qlora.py \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --cut_len 1536
```

### For High VRAM (20GB+)
```bash
python training/train_qlora.py \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --cut_len 2048
```

## Deployment

### 1. Local Deployment
```bash
# Set environment variables
export BASE_MODEL=microsoft/Phi-3-mini-4k-instruct
export ADAPTER_DIR=./yori_flirty_adapter
export MAX_TOKENS=256

# Start server
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment
```bash
# Update docker environment
echo "ADAPTER_DIR=./training/yori_flirty_adapter" >> .env.docker

# Restart containers
docker-compose down
docker-compose up --build
```

### 3. Production Deployment
```bash
# Copy adapter to production location
cp -r ./yori_flirty_adapter /app/models/production_adapter

# Update production environment
export ADAPTER_DIR=/app/models/production_adapter
export MAX_TOKENS=256

# Start with multiple workers
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## Testing Your Trained Model

### Basic Test
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi Yori!", "user_id": "test_user"}'
```

### Personality Test
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "You seem different today", "user_id": "test_user"}'
```

### Streaming Test
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me something interesting", "user_id": "test_user"}' \
  --no-buffer
```

## Troubleshooting

### Training Issues
```bash
# CUDA out of memory
python training/train_qlora.py --batch_size 1 --cut_len 1024

# No GPU detected
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.is_available())"

# Data loading errors
python -c "
import json
with open('./data/yori_train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Error on line {i+1}')
"
```

### Server Issues
```bash
# Adapter not loading
ls -la ./yori_flirty_adapter/
python -c "from peft import PeftModel; print('PEFT available')"

# Port conflicts
sudo lsof -i :8000
pkill -f uvicorn

# Memory issues
export MAX_TOKENS=128
free -h
```

## Performance Tips

### Faster Training
- Use `--packing=True` for shorter sequences
- Enable Flash Attention 2 on supported GPUs
- Use `--dataloader_num_workers=4` for faster data loading

### Better Quality
- Increase `--num_train_epochs` to 5-10
- Lower `--learning_rate` to 1e-4 for stable training
- Use larger `--cut_len` for longer context understanding

### Production Optimization
- Set `MAX_TOKENS=256` or lower for faster responses
- Use `temperature=0.8` for good personality balance
- Enable model quantization for memory efficiency

## Model Comparison

Test different adapters to find the best personality:

```bash
# Test base model (no adapter)
unset ADAPTER_DIR
python server/main.py &

# Test trained adapter
export ADAPTER_DIR=./yori_flirty_adapter
python server/main.py &

# Compare responses with same prompt
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How are you feeling today?", "user_id": "comparison_test"}'
```