#!/usr/bin/env python3

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", memory_dir: str = "./memory"):
        """Initialize the memory manager with FAISS vector store."""
        self.memory_dir = memory_dir
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.memories = []  # Store memory metadata
        self.user_memories = {}  # Track memories per user
        
        # Load existing memories
        self.load_memories()

    def _get_index_path(self) -> str:
        """Get the path to the FAISS index file."""
        return os.path.join(self.memory_dir, "memory_index.faiss")

    def _get_memories_path(self) -> str:
        """Get the path to the memories metadata file."""
        return os.path.join(self.memory_dir, "memories.pkl")

    def load_memories(self):
        """Load existing memories from disk."""
        index_path = self._get_index_path()
        memories_path = self._get_memories_path()
        
        try:
            if os.path.exists(index_path) and os.path.exists(memories_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load memories metadata
                with open(memories_path, 'rb') as f:
                    data = pickle.load(f)
                    self.memories = data.get('memories', [])
                    self.user_memories = data.get('user_memories', {})
                
                logger.info(f"Loaded {len(self.memories)} memories from disk")
            else:
                logger.info("No existing memories found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            # Reset to empty state if loading fails
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.memories = []
            self.user_memories = {}

    def save_memories(self):
        """Save memories to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self._get_index_path())
            
            # Save memories metadata
            with open(self._get_memories_path(), 'wb') as f:
                pickle.dump({
                    'memories': self.memories,
                    'user_memories': self.user_memories
                }, f)
            
            logger.info(f"Saved {len(self.memories)} memories to disk")
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")

    def add_memory(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new memory to the vector store.
        
        Args:
            user_id: The user ID to associate this memory with
            text: The text content to store and index
            metadata: Optional metadata dictionary
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([text])
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize for cosine similarity
            
            # Add to FAISS index
            self.index.add(embedding.astype(np.float32))
            
            # Store memory metadata
            memory = {
                'id': len(self.memories),
                'text': text,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.memories.append(memory)
            
            # Track user memories
            if user_id not in self.user_memories:
                self.user_memories[user_id] = []
            self.user_memories[user_id].append(memory['id'])
            
            # Auto-save periodically (every 10 memories)
            if len(self.memories) % 10 == 0:
                self.save_memories()
            
            logger.debug(f"Added memory for user {user_id}: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")

    def recall(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recall the most relevant memories for a query.
        
        Args:
            user_id: The user ID to filter memories by
            query: The query text to search for similar memories
            k: Number of top memories to return
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        try:
            if len(self.memories) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search for similar memories
            scores, indices = self.index.search(query_embedding.astype(np.float32), min(k * 2, len(self.memories)))
            
            # Filter by user and return top k
            relevant_memories = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.memories):  # Valid index
                    memory = self.memories[idx]
                    if (memory['user_id'] == user_id and 
                        score > 0.1 and  # Minimum similarity threshold
                        not memory.get('deleted', False)):  # Skip deleted memories
                        memory_copy = memory.copy()
                        memory_copy['similarity'] = float(score)
                        relevant_memories.append(memory_copy)
            
            # Sort by similarity and return top k
            relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
            return relevant_memories[:k]
            
        except Exception as e:
            logger.error(f"Error recalling memories: {str(e)}")
            return []

    def retrieve_memories(self, query: str, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility. Use recall() instead."""
        return self.recall(user_id, query, k)

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the number of memories for a specific user."""
        return len(self.user_memories.get(user_id, []))

    def clear_user_memories(self, user_id: str):
        """Clear all memories for a specific user."""
        try:
            if user_id in self.user_memories:
                memory_ids = self.user_memories[user_id]
                
                # Mark memories as deleted (we can't easily remove from FAISS index)
                for memory_id in memory_ids:
                    if memory_id < len(self.memories):
                        self.memories[memory_id]['deleted'] = True
                
                # Clear user memory tracking
                del self.user_memories[user_id]
                
                self.save_memories()
                logger.info(f"Cleared {len(memory_ids)} memories for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error clearing user memories: {str(e)}")

    def get_recent_memories(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent memories for a user."""
        try:
            if user_id not in self.user_memories:
                return []
            
            memory_ids = self.user_memories[user_id]
            recent_memories = []
            
            for memory_id in reversed(memory_ids[-limit:]):  # Get last N memory IDs
                if memory_id < len(self.memories):
                    memory = self.memories[memory_id]
                    if not memory.get('deleted', False):
                        recent_memories.append(memory)
            
            return recent_memories
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []

    def __del__(self):
        """Save memories when the object is destroyed."""
        try:
            self.save_memories()
        except Exception:
            pass  # Ignore errors during cleanup