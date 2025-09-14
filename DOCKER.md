# Docker Deployment Guide

This guide explains how to run Yori.chat using Docker with GPU support and PostgreSQL.

## Prerequisites

1. **Docker & Docker Compose**: Install Docker Desktop or Docker Engine
2. **NVIDIA Container Toolkit**: Required for GPU support
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **GPU with CUDA support**: NVIDIA GPU with at least 8GB VRAM recommended

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd yori.chat

# Copy environment template
cp .env.docker .env
# Edit .env file with your configurations
```

### 2. Build and Run
```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f yori-app
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Chat test
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Yori!", "user_id": "test_user"}'

# Streaming test
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "user_id": "test_user"}' \
  --no-buffer
```

## Service Details

### FastAPI Application (`yori-app`)
- **Port**: 8000
- **GPU Support**: NVIDIA CUDA 12.1
- **Memory**: Persistent storage in `/app/memory`
- **Models**: Cached in `/app/models`

### PostgreSQL Database (`postgres`)
- **Port**: 5432
- **Database**: `yori_db`
- **User**: `yori_user`
- **Extensions**: pgvector for embeddings
- **Data**: Persistent storage in `postgres_data` volume

### Volumes
- `postgres_data`: Database files
- `model_cache`: Hugging Face model cache
- `memory_data`: Application memory storage
- `logs_data`: Application logs

## Configuration

### Environment Variables (.env)
```bash
# Model settings
BASE_MODEL=microsoft/Phi-3-mini-4k-instruct
ADAPTER_DIR=  # Path to LoRA adapter (optional)
MAX_TOKENS=512

# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://yori_user:password@postgres:5432/yori_db

# Application
LOG_LEVEL=INFO
DEVICE=cuda  # or 'cpu' for CPU-only
```

### GPU Memory Requirements
- **Minimum**: 8GB VRAM (may need reduced batch sizes)
- **Recommended**: 16GB+ VRAM for comfortable training and inference
- **RTX 4090**: 24GB VRAM (ideal)

## Database Management

### Connect to PostgreSQL
```bash
# Using psql inside container
docker-compose exec postgres psql -U yori_user -d yori_db

# Or from host (if psql installed)
psql -h localhost -p 5432 -U yori_user -d yori_db
```

### View Memory Data
```sql
-- Check stored memories
SELECT id, user_id, text, created_at FROM memories ORDER BY created_at DESC LIMIT 10;

-- Check vector embeddings
SELECT id, user_id, embedding FROM memories WHERE user_id = 'test_user';

-- Search similar memories (example)
SELECT id, text, (embedding <=> '[0.1,0.2,...]'::vector) as distance 
FROM memories 
WHERE user_id = 'test_user' 
ORDER BY distance LIMIT 5;
```

## Training with Docker

### Train LoRA Adapter
```bash
# Execute training inside container
docker-compose exec yori-app python /app/training/train_qlora.py \
  --data_path /app/data/yori_train.jsonl \
  --output_dir /app/training/yori_adapter \
  --max_steps 100

# Update environment to use trained adapter
echo "ADAPTER_DIR=/app/training/yori_adapter" >> .env
docker-compose restart yori-app
```

## Monitoring & Debugging

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs yori-app
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f yori-app
```

### Check GPU Usage
```bash
# Inside container
docker-compose exec yori-app nvidia-smi

# From host
nvidia-smi
```

### Container Shell Access
```bash
# Access app container
docker-compose exec yori-app bash

# Access database container
docker-compose exec postgres bash
```

## Production Considerations

### Security
1. Change default passwords in `.env`
2. Use secrets management for production
3. Enable SSL/TLS (uncomment nginx service)
4. Restrict database access

### Performance
1. Tune PostgreSQL settings for your workload
2. Configure connection pooling
3. Set up monitoring (Prometheus/Grafana)
4. Use GPU with sufficient VRAM

### Scaling
1. Add Redis for session storage
2. Use load balancer (nginx) for multiple app instances
3. Implement database read replicas
4. Consider Kubernetes for orchestration

## Backup & Recovery

### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U yori_user yori_db > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U yori_user yori_db < backup.sql
```

### Volume Backup
```bash
# Backup all data
docker run --rm -v yori_postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres_backup.tar.gz -C /data .

docker run --rm -v yori_model_cache:/data -v $(pwd):/backup alpine \
  tar czf /backup/models_backup.tar.gz -C /data .
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Verify Docker GPU support
docker-compose exec yori-app python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Monitor container memory
docker stats

# Check GPU memory
docker-compose exec yori-app nvidia-smi

# Clear model cache
docker-compose exec yori-app rm -rf /app/models/*
```

### Database Connection Issues
```bash
# Check database status
docker-compose exec postgres pg_isready -U yori_user

# Test connection from app
docker-compose exec yori-app python -c "
import psycopg2
conn = psycopg2.connect('postgresql://yori_user:password@postgres:5432/yori_db')
print('Database connected successfully')
"
```

### Port Conflicts
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Use different ports
# Edit docker-compose.yml ports section: "8001:8000"
```

## Stopping Services

```bash
# Stop services (keeps volumes)
docker-compose down

# Stop and remove everything including volumes
docker-compose down -v

# Remove images as well
docker-compose down --rmi all -v
```