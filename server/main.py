#!/usr/bin/env python3

import os
import torch
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from dotenv import load_dotenv
import logging
from memory import MemoryManager
from collections import defaultdict, deque
from typing import Dict, List, Generator
from threading import Thread
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yori Chat API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"

class ChatResponse(BaseModel):
    response: str

class StreamRequest(BaseModel):
    message: str
    user_id: str = "default"

class YoriModel:
    def __init__(self):
        self.base_model = "microsoft/Phi-3-mini-4k-instruct"  # Fixed base model
        self.adapter_dir = os.getenv("ADAPTER_DIR", None)
        self.max_tokens = int(os.getenv("MAX_TOKENS", "256"))  # Default 256 tokens
        self.model = None
        self.tokenizer = None
        self.memory_manager = MemoryManager()
        # Short-term memory: keep last 8 turns per user
        self.short_term_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=8))
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Load LoRA adapter if specified
        if self.adapter_dir and os.path.exists(self.adapter_dir):
            logger.info(f"Loading LoRA adapter from: {self.adapter_dir}")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
                logger.info("LoRA adapter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter: {e}")
                logger.info("Continuing with base model only")
        elif self.adapter_dir:
            logger.warning(f"Adapter directory not found: {self.adapter_dir}")
            logger.info("Continuing with base model only")

        logger.info("Model loaded successfully")

    def generate_response(self, message: str, user_id: str) -> str:
        """Generate response using the model with memory context."""
        
        # Check for self-harm content
        is_flagged, safe_response = self.moderate(message)
        if is_flagged:
            # Still add to short-term memory for context
            self.short_term_memory[user_id].append(f"User: {message}")
            self.short_term_memory[user_id].append(f"Yori: {safe_response}")
            return safe_response
        
        # Get top 3 recalled facts from long-term memory
        recalled_facts = self.memory_manager.recall(user_id, message, k=3)
        facts_context = ""
        if recalled_facts:
            facts_list = [f"- {fact['text']}" for fact in recalled_facts]
            facts_context = f"\n\nKnown facts about the user:\n" + "\n".join(facts_list)
        
        # Get short-term conversation history
        conversation_history = ""
        if user_id in self.short_term_memory:
            history_turns = list(self.short_term_memory[user_id])
            if history_turns:
                conversation_history = "\n\nRecent conversation:\n" + "\n".join(history_turns)
        
        # Format the prompt with new persona
            system_prompt = (
    "You are Yori, a flirty, playful AI companion. "
    "Reply warmly and naturally, but keep your messages short like a text — 1–2 sentences max."
    "Be concise and avoid rambling."
    )
        
        prompt = f"<|system|>\n{system_prompt}{facts_context}{conversation_history}<|end|>\n<|user|>\n{message}<|end|>\n<|assistant|>\n"

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=0.8,  # Slightly higher for more personality
                do_sample=True,
                top_p=0.9,
                top_k=40,  # Reduced for more focused responses
                repetition_penalty=1.05,  # Reduced to allow natural repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = response.strip()

        # Add to short-term memory
        self.short_term_memory[user_id].append(f"User: {message}")
        self.short_term_memory[user_id].append(f"Yori: {response}")

        # Store important information in long-term memory
        # Only store if the user shares personal information or facts about themselves
        if self._contains_personal_info(message):
            self.memory_manager.add_memory(user_id, message)

        return response

    def generate_response_stream(self, message: str, user_id: str) -> Generator[str, None, None]:
        """Generate response using the model with streaming output."""
        
        # Check for self-harm content
        is_flagged, safe_response = self.moderate(message)
        if is_flagged:
            # Still add to short-term memory for context
            self.short_term_memory[user_id].append(f"User: {message}")
            self.short_term_memory[user_id].append(f"Yori: {safe_response}")
            yield safe_response
            return
        
        # Get top 3 recalled facts from long-term memory
        recalled_facts = self.memory_manager.recall(user_id, message, k=3)
        facts_context = ""
        if recalled_facts:
            facts_list = [f"- {fact['text']}" for fact in recalled_facts]
            facts_context = f"\n\nKnown facts about the user:\n" + "\n".join(facts_list)
        
        # Get short-term conversation history
        conversation_history = ""
        if user_id in self.short_term_memory:
            history_turns = list(self.short_term_memory[user_id])
            if history_turns:
                conversation_history = "\n\nRecent conversation:\n" + "\n".join(history_turns)
        
        # Format the prompt with new persona
        system_prompt = "You are Yori, a warm, playful, and conversational AI companion. Keep your replies concise and natural - just 1 to 4 sentences. Be friendly and engaging without writing long paragraphs."
        
        prompt = f"<|system|>\n{system_prompt}{facts_context}{conversation_history}<|end|>\n<|user|>\n{message}<|end|>\n<|assistant|>\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Set up streaming
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=60.0, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_tokens,
            "temperature": 0.8,  # Slightly higher for more personality
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 40,  # Reduced for more focused responses
            "repetition_penalty": 1.05,  # Reduced to allow natural repetition
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        full_response = ""
        for new_token in streamer:
            full_response += new_token
            yield new_token
        
        thread.join()
        
        # Add to short-term memory
        self.short_term_memory[user_id].append(f"User: {message}")
        self.short_term_memory[user_id].append(f"Yori: {full_response}")

        # Store important information in long-term memory
        if self._contains_personal_info(message):
            self.memory_manager.add_memory(user_id, message)

    def _contains_personal_info(self, message: str) -> bool:
        """Check if message contains personal information worth storing."""
        personal_indicators = [
            "my name is", "i am", "i'm", "i work", "i live", "i like", "i love", 
            "i hate", "i enjoy", "my favorite", "i have", "i own", "i study",
            "i'm from", "i was born", "my birthday", "my age", "my hobby",
            "my job", "my family", "my pet", "my friend"
        ]
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in personal_indicators)

    def moderate(self, text: str) -> tuple[bool, str]:
        """Moderate text for self-harm content.
        
        Returns:
            tuple: (is_flagged, safe_response_if_flagged)
        """
        # Self-harm keywords and phrases
        self_harm_patterns = [
            r'\b(kill|hurt|harm)\s+(myself|me)\b',
            r'\b(suicide|suicidal)\b',
            r'\bend\s+my\s+life\b',
            r'\bdon\'?t\s+want\s+to\s+live\b',
            r'\bhate\s+myself\b',
            r'\bwant\s+to\s+die\b',
            r'\bcut\s+(myself|me)\b',
            r'\bhurt\s+(myself|me)\b',
            r'\bkill\s+(myself|me)\b'
        ]
        
        text_lower = text.lower()
        
        for pattern in self_harm_patterns:
            if re.search(pattern, text_lower):
                safe_response = (
                    "I'm really concerned about what you're going through right now. "
                    "Your feelings matter, and there are people who want to help. "
                    "Please consider reaching out to a crisis helpline like 988 (Suicide & Crisis Lifeline) "
                    "or text HOME to 741741 (Crisis Text Line). "
                    "You don't have to face this alone, and there are trained professionals "
                    "who can provide the support you deserve."
                )
                return True, safe_response
        
        return False, ""

yori = YoriModel()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": yori.base_model}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        response = yori.generate_response(request.message, request.user_id)
        return ChatResponse(response=response)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/stream")
async def stream_chat(request: StreamRequest):
    """Streaming chat endpoint using Server Sent Events."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        async def generate_stream():
            """Generate SSE stream."""
            try:
                for token in yori.generate_response_stream(request.message, request.user_id):
                    # Format as Server-Sent Event
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming client
                
                # Send end signal
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': 'Generation failed'})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Yori Chat API",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "stream": "POST /stream"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)