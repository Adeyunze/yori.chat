# yori.chat - AI Companion

A minimal AI companion built with Microsoft Phi-3-mini-4k-instruct, featuring LoRA fine-tuning and long-term memory.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Train Custom Adapter (Optional)
```bash
python training/train_qlora.py
```

### 4. Run Server
```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Test API
```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Yori!"}'
```

## Docker Deployment

```bash
docker build -t yori-chat .
docker run -p 8000:8000 yori-chat
```

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Chat with Yori
  - Body: `{"message": "your message"}`
  - Response: `{"response": "yori's response"}`

## Project Structure

```
yori.chat/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── data/
│   └── yori_train.jsonl
├── training/
│   └── train_qlora.py
└── server/
    ├── main.py
    └── memory.py
```