# RunPod Deployment Guide for Yori.chat

This guide explains how to deploy and train Yori.chat on RunPod GPU infrastructure. Yori.chat is an AI companion powered by teknium/OpenHermes-2.5-Mistral-7B with QLoRA fine-tuning, featuring memory management and a Next.js frontend.

## 1. Launch GPU Pod

### Step 1: Create RunPod Account

1. Go to [runpod.io](https://runpod.io) and sign up
2. Add payment method and credits to your account
3. Navigate to "Pods" section

### Step 2: Launch Pod with PyTorch Template

1. Click "Deploy" → "GPU Pod"
2. **Template Selection:**

   - Search for "PyTorch" template
   - Select "RunPod PyTorch 2.1" or latest stable version
   - This includes CUDA, PyTorch, and common ML libraries pre-installed

3. **GPU Selection:**

   - **Recommended:** RTX 4090 (24GB VRAM) for training and inference
   - **Budget option:** RTX 3090 (24GB VRAM)
   - **Minimum:** RTX 3080 (10GB VRAM) - may need reduced batch size for training
   - **Inference only:** RTX 3060 (12GB VRAM) or RTX 4060 Ti (16GB VRAM)

4. **Storage Configuration:**
   - **Container Disk:** 80GB minimum (for models and dependencies)
   - **Volume Disk:** 50GB persistent storage (recommended)
   - Enable "Persistent Volume" to save models and memory data between sessions

### Step 3: Configure Pod Settings

```
Pod Name: yori-chat-deployment
Container Disk: 80 GB
Volume Disk: 50 GB (persistent)
Expose HTTP Ports: 8000, 3000
Expose TCP Ports: 22 (for SSH)
```

5. Click "Deploy On-Demand" (or "Spot" for cheaper but interruptible instances)

## 2. Connect to Your Pod

### Option A: Web Terminal (Easiest)

1. Wait for pod status to show "Running"
2. Click "Connect" → "Start Web Terminal"
3. Access Jupyter-style terminal in browser

### Option B: SSH Connection

1. Click "Connect" → "TCP Port Mappings"
2. Note the SSH connection details (usually port 22)
3. Use provided SSH command:

```bash
ssh root@<pod-id>.tcpport.22.runpod.io
# Enter password when prompted
```

## 3. Setup and Training Commands

### Step 1: Clone Repository

```bash
# Navigate to workspace (if using persistent volume)
cd /workspace

# Clone the repository
git clone https://github.com/your-username/yori.chat.git
cd yori.chat

# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install Python requirements
pip install -r requirements.txt

# Install Node.js for frontend (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify Python installations
python -c "import torch, transformers, peft, trl, fastapi, uvicorn; print('All Python packages installed successfully')"

# Verify Node.js installation
node --version
npm --version
```

### Step 4: Run Training

```bash
# Navigate to training directory
cd training

# Start training (adjust parameters as needed)
python train_qlora.py \
  --model_name teknium/OpenHermes-2.5-Mistral-7B \
  --data_path ../data/yori_train.jsonl \
  --out_dir ./yori_adapter \
  --cut_len 2048 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --warmup_steps 10 \
  --logging_steps 5 \
  --save_steps 50

# Training will save adapter to ./yori_adapter/
# Check training progress with: tail -f logs/training.log
```

### Step 5: Start the Server

```bash
# Navigate back to project root
cd /workspace/yori.chat

# Activate virtual environment if not already active
source venv/bin/activate

# Set environment variables
export BASE_MODEL=teknium/OpenHermes-2.5-Mistral-7B
export ADAPTER_DIR=training/yori_adapter
export MAX_TOKENS=512
export MEMORY_DIR=/workspace/yori.chat/memory
export EMBEDDING_MODEL=all-MiniLM-L6-v2

# Start the server
cd server
uvicorn main:app --host 0.0.0.0 --port 8000

# Server will start and show:
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Deploy the Frontend (Optional)

```bash
# Navigate to frontend directory
cd /workspace/yori.chat/frontend-next

# Install frontend dependencies
npm install

# Build the frontend for production
npm run build

# Start the frontend server
npm start

# Frontend will be available on port 3000
# Access via: https://your-pod-id-3000.proxy.runpod.net
```

## 4. Expose and Test the API

### Step 1: Configure Port Exposure

1. In RunPod UI, go to your pod
2. Click "Connect" → "HTTP Service [Port 8000]"
3. Note the public URL (e.g., `https://pod-id-8000.proxy.runpod.net`)

### Step 2: Test with curl

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Chat endpoint (non-streaming)
curl -X POST https://09sldnal60n6gk-19123.proxy.runpod.net/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, I am Yunus", "user_id": "test_user"}'

# Streaming endpoint (Server-Sent Events)
curl -X POST https://your-pod-id-8000.proxy.runpod.net/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "user_id": "test_user"}' \
  -N

# Root endpoint (API information)
curl https://your-pod-id-8000.proxy.runpod.net/
```

## 5. Memory System

Yori.chat includes a sophisticated memory management system that learns and remembers information about users:

### Features

- **Long-term Memory**: Stores important user information using FAISS vector search
- **Short-term Memory**: Maintains conversation context (last 8 turns)
- **Personalization**: Learns user preferences, interests, and important details
- **Automatic Storage**: Detects and stores personal information automatically

### Memory Storage

```bash
# Memory data is stored in the persistent volume
ls -la /workspace/yori.chat/memory/
# Contains: memory_index.faiss, memories.pkl

# Memory persists between pod restarts when using persistent volume
```

### Memory Management

- Memories are automatically saved every 10 new entries
- Vector embeddings use `all-MiniLM-L6-v2` model
- Similarity search with configurable thresholds
- User-specific memory isolation

## 6. Stop Pod to Avoid Charges

### Important: Always stop your pod when done!

1. Go to RunPod dashboard
2. Find your pod in the list
3. Click "Stop" or "Terminate"
4. **Stop:** Preserves persistent volume, charges storage only
5. **Terminate:** Deletes everything, no further charges

## Troubleshooting

### CUDA Issues

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If CUDA not found, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test CUDA in Python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Memory Issues

```bash
# If training fails with OOM (Out of Memory):
python train_qlora.py \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 256

# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear GPU cache in Python
python -c "import torch; torch.cuda.empty_cache()"
```

### Networking Issues

```bash
# If server won't start:
# Check if port is already in use
netstat -tulpn | grep 8000

# Kill existing processes on port 8000
sudo lsof -t -i tcp:8000 | xargs kill -9

# Test local connectivity
curl localhost:8000/health

# If public URL not working, check RunPod proxy settings
# Ensure HTTP Port 8000 is properly exposed in pod configuration
```

### Package Installation Issues

```bash
# If pip install fails, try with --no-cache-dir
pip install -r requirements.txt --no-cache-dir

# For specific package conflicts:
pip install --upgrade --force-reinstall package_name

# If bitsandbytes fails, install pre-compiled version:
pip install bitsandbytes --prefer-binary
```

### Training Issues

```bash
# If training crashes, reduce parameters:
python train_qlora.py \
  --model_name teknium/OpenHermes-2.5-Mistral-7B \
  --num_train_epochs 1 \
  --batch_size 1 \
  --cut_len 1024 \
  --gradient_accumulation_steps 8

# Monitor training progress:
tail -f logs/training.log  # if logging to file

# Check GPU memory usage during training:
watch -n 1 nvidia-smi
```

### Memory System Issues

```bash
# If memory system fails to load:
# Check if memory directory exists and has proper permissions
ls -la /workspace/yori.chat/memory/
chmod 755 /workspace/yori.chat/memory/

# Reset memory system (will lose all stored memories):
rm -rf /workspace/yori.chat/memory/*
mkdir -p /workspace/yori.chat/memory

# Check memory system logs:
grep -i memory /workspace/yori.chat/logs/*.log
```

### Frontend Issues

```bash
# If frontend fails to build:
cd /workspace/yori.chat/frontend-next
rm -rf node_modules package-lock.json
npm install

# If frontend won't start:
# Check if port 3000 is available
netstat -tulpn | grep 3000

# Kill existing processes on port 3000
sudo lsof -t -i tcp:3000 | xargs kill -9
```

### File Persistence

```bash
# To ensure files persist between pod restarts:
# Always work in /workspace if you have persistent volume
cd /workspace

# Check persistent volume mount
df -h | grep workspace

# Backup important files to persistent storage
cp -r /tmp/important_files /workspace/backup/

# Verify memory and model persistence
ls -la /workspace/yori.chat/memory/
ls -la /workspace/yori.chat/training/yori_adapter/
```

## Cost Optimization Tips

1. **Use Spot Instances:** 50-80% cheaper but can be interrupted
2. **Stop Pods:** Always stop when not actively using
3. **Monitor Usage:** Check RunPod dashboard for spending
4. **Persistent Storage:** Only pay for storage when pod is stopped
5. **GPU Selection:** Use minimum GPU that meets your needs
6. **Memory Management:** Memory data persists between sessions, reducing re-training needs
7. **Frontend Caching:** Use CDN or caching for frontend assets

## Estimated Costs (as of 2024)

- **RTX 4090:** ~$0.50-0.70/hour
- **RTX 3090:** ~$0.40-0.60/hour
- **Persistent Storage:** ~$0.10/GB/month
- **Training Time:** ~45-90 minutes for 3 epochs (varies by dataset size)
- **Inference Only:** ~$0.30-0.50/hour (no training needed after initial setup)

## Key Features

### AI Model

- **Base Model:** teknium/OpenHermes-2.5-Mistral-7B (7B parameters)
- **Fine-tuning:** QLoRA with 4-bit quantization
- **Memory:** 6GB VRAM minimum for inference, 12GB+ for training

### Memory System

- **Vector Search:** FAISS-based similarity search
- **Embeddings:** all-MiniLM-L6-v2 (384 dimensions)
- **Persistence:** Automatic saving and loading
- **Personalization:** User-specific memory isolation

### API Endpoints

- **POST /chat:** Non-streaming chat responses
- **POST /stream:** Server-Sent Events streaming
- **GET /health:** Health check and model info
- **GET /:** API documentation

### Frontend

- **Framework:** Next.js 14 with TypeScript
- **Styling:** Tailwind CSS
- **Features:** Real-time chat, typing indicators, responsive design

Remember to always stop your pod when finished to avoid unnecessary charges!
