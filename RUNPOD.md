# RunPod Deployment Guide for Yori.chat

This guide explains how to deploy and train Yori.chat on RunPod GPU infrastructure.

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
   - **Recommended:** RTX 4090 (24GB VRAM) for training
   - **Budget option:** RTX 3090 (24GB VRAM)
   - **Minimum:** RTX 3080 (10GB VRAM) - may need reduced batch size

4. **Storage Configuration:**
   - **Container Disk:** 50GB minimum
   - **Volume Disk:** 20GB persistent storage (recommended)
   - Enable "Persistent Volume" to save models between sessions

### Step 3: Configure Pod Settings
```
Pod Name: yori-chat-training
Container Disk: 50 GB
Volume Disk: 20 GB (persistent)
Expose HTTP Ports: 8000
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
# Install requirements
pip install -r requirements.txt

# Verify installations
python -c "import torch, transformers, peft, trl; print('All packages installed successfully')"
```

### Step 4: Run Training
```bash
# Navigate to training directory
cd training

# Start training (adjust parameters as needed)
python train_qlora.py \
  --model_name microsoft/Phi-3-mini-4k-instruct \
  --data_path ../data/yori_train.jsonl \
  --output_dir ./yori_adapter \
  --max_steps 100 \
  --batch_size 2 \
  --learning_rate 2e-4

# Training will save adapter to ./yori_adapter/
```

### Step 5: Start the Server
```bash
# Navigate back to project root
cd /workspace/yori.chat

# Activate virtual environment if not already active
source venv/bin/activate

# Set environment variables
export BASE_MODEL=microsoft/Phi-3-mini-4k-instruct
export ADAPTER_DIR=training/yori_adapter
export MAX_TOKENS=512

# Start the server
cd server
uvicorn main:app --host 0.0.0.0 --port 8000

# Server will start and show:
# INFO: Uvicorn running on http://0.0.0.0:8000
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

# Chat endpoint
curl -X POST https://27u5e78ip5v3e0-8000.proxy.runpod.net/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, I am testing the deployment!", "user_id": "test_user"}'

# Streaming endpoint
curl -X POST https://27u5e78ip5v3e0-8000.proxy.runpod.net/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "user_id": "test_user"}' \
```

## 5. Stop Pod to Avoid Charges

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
  --max_steps 50 \
  --batch_size 1 \
  --max_seq_length 256 \
  --gradient_accumulation_steps 4

# Monitor training progress:
tail -f logs/training.log  # if logging to file
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
```

## Cost Optimization Tips

1. **Use Spot Instances:** 50-80% cheaper but can be interrupted
2. **Stop Pods:** Always stop when not actively using
3. **Monitor Usage:** Check RunPod dashboard for spending
4. **Persistent Storage:** Only pay for storage when pod is stopped
5. **GPU Selection:** Use minimum GPU that meets your needs

## Estimated Costs (as of 2024)
- **RTX 4090:** ~$0.50-0.70/hour
- **RTX 3090:** ~$0.40-0.60/hour  
- **Persistent Storage:** ~$0.10/GB/month
- **Training Time:** ~15-30 minutes for 100 steps

Remember to always stop your pod when finished to avoid unnecessary charges!