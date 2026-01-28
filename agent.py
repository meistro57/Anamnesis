"""
Anamnesis - Agent API / Integration Layer

This module provides a high-level API for integrating Anamnesis
into existing systems like Chat Bridge, Eli GPT, or other
AI assistants.

Key Features:
1. Simple record/retrieve/feedback interface
2. Optional embedding support (OpenAI, local models)
3. Async-ready design
4. Event hooks for custom integrations
"""

import logging
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

# Configure module logger
logger = logging.getLogger(__name__)

try:
    from .core import (
        MemRLStore, TwoPhaseRetriever, QLearner,
        EpisodicMemory, create_memory_id
    )
except ImportError:
    # Allow running as standalone script
    from core import (
        MemRLStore, TwoPhaseRetriever, QLearner,
        EpisodicMemory, create_memory_id
    )


@dataclass
class RetrievalResult:
    """Result from memory retrieval"""
    memory_id: str
    query_summary: str
    action_taken: str
    q_value: float
    similarity: float
    success: bool
    metadata: Dict[str, Any]


@dataclass 
class FeedbackResult:
    """Result from providing feedback"""
    memory_id: str
    old_q: float
    new_q: float
    improvement: float


class AnamnesisAgent:
    """
    High-level agent interface for Anamnesis

    This is the main class you'd integrate into Chat Bridge,
    Eli GPT, or other systems.

    Example usage:

        agent = AnamnesisAgent("./memories.db")

        # When user asks a question
        context = agent.get_context("How do I meditate?", task_type="spiritual")

        # After responding, record the interaction
        memory_id = agent.record_interaction(
            task_type="spiritual",
            query="How do I meditate?",
            response="Start with 5 minutes of breath awareness...",
            context="User is new to meditation",
            success=True
        )

        # When user gives feedback (thumbs up/down)
        agent.provide_feedback(memory_id, positive=True)
    """

    # Validation constants
    MAX_QUERY_LENGTH = 10000
    MAX_RESPONSE_LENGTH = 50000
    MAX_TASK_TYPE_LENGTH = 100
    
    def __init__(self, 
                 db_path: str = "memrl_agent.db",
                 embedding_fn: Optional[Callable[[str], List[float]]] = None,
                 learning_rate: float = 0.15,
                 on_memory_stored: Optional[Callable[[EpisodicMemory], None]] = None,
                 on_feedback: Optional[Callable[[str, float, float], None]] = None):
        """
        Initialize MemRL Agent
        
        Args:
            db_path: Path to SQLite database
            embedding_fn: Optional function to compute embeddings
            learning_rate: How fast Q-values adapt (0.0-1.0)
            on_memory_stored: Callback when memory is stored
            on_feedback: Callback when feedback is received (id, old_q, new_q)
        """
        self.store = MemRLStore(db_path)
        self.retriever = TwoPhaseRetriever(self.store, embedding_fn)
        self.learner = QLearner(self.store, learning_rate=learning_rate)
        
        self.embedding_fn = embedding_fn
        self.on_memory_stored = on_memory_stored
        self.on_feedback = on_feedback

        # Track current interaction for feedback
        self._current_interaction_memories: List[str] = []

    def _validate_query(self, query: str) -> str:
        """Validate and sanitize query input"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        query = query.strip()
        if len(query) > self.MAX_QUERY_LENGTH:
            logger.warning(f"Query truncated from {len(query)} to {self.MAX_QUERY_LENGTH} chars")
            query = query[:self.MAX_QUERY_LENGTH]
        return query

    def _validate_task_type(self, task_type: Optional[str]) -> Optional[str]:
        """Validate task type input"""
        if task_type is None:
            return None
        if not isinstance(task_type, str):
            raise ValueError("task_type must be a string")
        task_type = task_type.strip()
        if len(task_type) > self.MAX_TASK_TYPE_LENGTH:
            raise ValueError(f"task_type must be <= {self.MAX_TASK_TYPE_LENGTH} characters")
        return task_type

    def _validate_rating(self, rating: Optional[float]) -> Optional[float]:
        """Validate rating is within bounds"""
        if rating is None:
            return None
        if not isinstance(rating, (int, float)):
            raise ValueError("rating must be a number")
        if rating < -1.0 or rating > 1.0:
            raise ValueError("rating must be between -1.0 and 1.0")
        return float(rating)
    
    def get_context(self,
                    query: str,
                    task_type: Optional[str] = None,
                    num_results: int = 3,
                    min_q: float = -0.3) -> List[RetrievalResult]:
        """
        Retrieve relevant memories to provide context for responding

        Args:
            query: The user's question/request
            task_type: Category filter (e.g., "code_help", "spiritual")
            num_results: How many memories to retrieve (1-100)
            min_q: Minimum Q-value threshold (-1.0 to 1.0)

        Returns:
            List of relevant memories with their Q-values

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        query = self._validate_query(query)
        task_type = self._validate_task_type(task_type)

        if not isinstance(num_results, int) or num_results < 1 or num_results > 100:
            raise ValueError("num_results must be an integer between 1 and 100")

        if not isinstance(min_q, (int, float)) or min_q < -1.0 or min_q > 1.0:
            raise ValueError("min_q must be a number between -1.0 and 1.0")

        results = self.retriever.retrieve(
            query,
            task_type=task_type,
            phase_a_k=num_results * 4,  # Get more candidates for filtering
            phase_b_n=num_results,
            min_q_threshold=min_q
        )
        
        # Track for feedback
        self._current_interaction_memories = [mem.id for mem, _ in results]
        
        return [
            RetrievalResult(
                memory_id=mem.id,
                query_summary=mem.query_summary,
                action_taken=mem.action_taken,
                q_value=mem.q_value,
                similarity=sim,
                success=mem.success,
                metadata=mem.metadata
            )
            for mem, sim in results
        ]
    
    def format_context_for_prompt(self, 
                                   results: List[RetrievalResult],
                                   max_chars: int = 2000) -> str:
        """
        Format retrieved memories for injection into an LLM prompt
        
        Args:
            results: Retrieved memories
            max_chars: Maximum characters to return
        
        Returns:
            Formatted string suitable for prompt injection
        """
        if not results:
            return ""
        
        lines = ["<relevant_past_experiences>"]
        
        char_count = len(lines[0])
        
        for i, r in enumerate(results):
            entry = f"""
<experience utility="{r.q_value:.2f}" success="{r.success}">
  <prior_query>{r.query_summary}</prior_query>
  <action_taken>{r.action_taken}</action_taken>
</experience>"""
            
            if char_count + len(entry) > max_chars:
                break
            
            lines.append(entry)
            char_count += len(entry)
        
        lines.append("</relevant_past_experiences>")
        
        return "\n".join(lines)
    
    def record_interaction(self,
                           task_type: str,
                           query: str,
                           response: str,
                           context: str = "",
                           outcome: str = "",
                           success: bool = True,
                           initial_reward: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a new interaction as episodic memory

        Args:
            task_type: Category (e.g., "code_help", "creative", "spiritual")
            query: User's query/request
            response: What was provided
            context: Additional context
            outcome: Description of what happened
            success: Whether it worked
            initial_reward: Initial Q-value (-1 to 1), defaults based on success
            metadata: Additional searchable metadata

        Returns:
            Memory ID for later feedback

        Raises:
            ValueError: If required inputs are invalid
        """
        # Validate inputs
        task_type = self._validate_task_type(task_type)
        if not task_type:
            raise ValueError("task_type is required and cannot be empty")

        query = self._validate_query(query)

        if not response or not isinstance(response, str):
            raise ValueError("response must be a non-empty string")
        response = response.strip()
        if len(response) > self.MAX_RESPONSE_LENGTH:
            logger.warning(f"Response truncated from {len(response)} to {self.MAX_RESPONSE_LENGTH} chars")
            response = response[:self.MAX_RESPONSE_LENGTH]

        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")

        initial_reward = self._validate_rating(initial_reward)
        if initial_reward is None:
            initial_reward = 0.5 if success else -0.3
        
        # Compute embedding if available
        embedding = None
        if self.embedding_fn:
            embed_text = f"{query} {response[:500]}"
            embedding = self.embedding_fn(embed_text)
        
        memory = EpisodicMemory(
            id=create_memory_id(query, response),
            task_type=task_type,
            query_summary=query[:500],  # Truncate for storage
            context_summary=context[:500],
            action_taken=response[:1000],
            outcome_description=outcome[:500],
            success=success,
            reward=initial_reward,
            q_value=initial_reward,
            embedding=embedding,
            retrieval_count=0,
            last_retrieved=None,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.store.store_memory(memory)
        
        if self.on_memory_stored:
            self.on_memory_stored(memory)
        
        return memory.id
    
    def provide_feedback(self,
                         memory_id: Optional[str] = None,
                         positive: bool = True,
                         rating: Optional[float] = None,
                         reason: str = "") -> Optional[FeedbackResult]:
        """
        Provide feedback on a memory (or the most recent interaction)

        Args:
            memory_id: Specific memory to update, or None for most recent
            positive: Simple thumbs up/down
            rating: Fine-grained rating (-1 to 1), overrides positive
            reason: Why this feedback was given

        Returns:
            FeedbackResult with Q-value changes, or None if memory not found

        Raises:
            ValueError: If rating is out of bounds
        """
        # Validate rating if provided
        rating = self._validate_rating(rating)

        if memory_id is None:
            if not self._current_interaction_memories:
                logger.warning("No current interaction memories to provide feedback for")
                return None
            memory_id = self._current_interaction_memories[0]

        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        memory = self.store.get_memory(memory_id)
        if not memory:
            logger.warning(f"Memory {memory_id} not found for feedback")
            return None

        # Determine reward signal
        if rating is not None:
            reward = rating
        else:
            reward = 0.8 if positive else -0.5
        
        old_q = memory.q_value
        new_q = self.learner.update_from_feedback(memory_id, reward, reason)
        
        if self.on_feedback:
            self.on_feedback(memory_id, old_q, new_q)
        
        return FeedbackResult(
            memory_id=memory_id,
            old_q=old_q,
            new_q=new_q,
            improvement=new_q - old_q
        )
    
    def bulk_feedback(self,
                      task_success: bool,
                      explicit_rating: Optional[float] = None) -> Dict[str, float]:
        """
        Provide feedback for all memories used in current interaction

        This is useful when you want to credit all retrieved memories
        based on overall task outcome.

        Args:
            task_success: Did the overall task succeed?
            explicit_rating: Optional user rating (-1 to 1)

        Returns:
            Dict mapping memory_id to new Q-value

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(task_success, bool):
            raise ValueError("task_success must be a boolean")

        explicit_rating = self._validate_rating(explicit_rating)

        if not self._current_interaction_memories:
            logger.debug("No current interaction memories for bulk feedback")
            return {}

        return self.learner.batch_update(
            self._current_interaction_memories,
            task_success,
            explicit_rating
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        return self.store.get_stats()

    def get_memory_q_value(self, memory_id: str) -> Optional[float]:
        """
        Get the current Q-value for a specific memory

        Args:
            memory_id: ID of the memory to query

        Returns:
            Q-value of the memory, or None if memory doesn't exist
        """
        memory = self.store.get_memory(memory_id)
        if memory:
            return memory.q_value
        return None
    
    def get_top_memories(self, 
                         n: int = 10,
                         task_type: Optional[str] = None) -> List[RetrievalResult]:
        """Get top memories by Q-value (most useful)"""
        memories = self.store.get_top_by_q_value(n, task_type)
        
        return [
            RetrievalResult(
                memory_id=mem.id,
                query_summary=mem.query_summary,
                action_taken=mem.action_taken,
                q_value=mem.q_value,
                similarity=1.0,  # Not from search
                success=mem.success,
                metadata=mem.metadata
            )
            for mem in memories
        ]
    
    def run_maintenance(self, 
                        decay_unused_days: int = 30,
                        decay_factor: float = 0.95) -> Dict[str, int]:
        """
        Run maintenance tasks:
        - Decay Q-values of unused memories
        
        Returns:
            Stats about maintenance performed
        """
        decayed = self.learner.decay_unused(decay_factor, decay_unused_days)
        
        return {
            "memories_decayed": decayed
        }


# ============================================
# Optional: Embedding Providers
# ============================================

def create_openai_embedding_fn(api_key: str, model: str = "text-embedding-3-small"):
    """Create embedding function using OpenAI API"""
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    def embed(text: str) -> List[float]:
        response = client.embeddings.create(
            model=model,
            input=text[:8000]  # Truncate to fit
        )
        return response.data[0].embedding
    
    return embed


def create_sentence_transformer_fn(model_name: str = "all-MiniLM-L6-v2"):
    """Create embedding function using sentence-transformers (local)"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    
    def embed(text: str) -> List[float]:
        return model.encode(text).tolist()
    
    return embed


# ============================================
# Demo / Integration Example
# ============================================

def demo_integration():
    """
    Demonstrate how Anamnesis would integrate with a chat system
    """
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("Anamnesis Integration Demo")
    logger.info("=" * 60)

    # Initialize agent
    agent = AnamnesisAgent("./integration_demo.db")

    # Simulate a conversation flow

    # 1. User asks a question
    user_query = "How can I quiet my mind during meditation?"
    logger.info(f"User: {user_query}")

    # 2. Check for relevant past experiences
    context = agent.get_context(user_query, task_type="spiritual")

    if context:
        logger.info("Retrieved context from past experiences:")
        for c in context:
            logger.info(f"   Q={c.q_value:.2f}: {c.action_taken[:60]}...")

        # This would be injected into the LLM prompt
        prompt_context = agent.format_context_for_prompt(context)
        logger.info(f"Context for LLM prompt:\n{prompt_context}")
    else:
        logger.info("No relevant past experiences found")

    # 3. Generate response (simulated)
    assistant_response = """Try the 'noting' technique: when thoughts arise,
simply label them as 'thinking' and return to your breath.
Don't fight the thoughts - acknowledge and release them."""

    logger.info(f"Assistant: {assistant_response}")

    # 4. Record this interaction
    memory_id = agent.record_interaction(
        task_type="spiritual",
        query=user_query,
        response=assistant_response,
        context="User learning meditation basics",
        success=True,
        metadata={"topic": "meditation", "technique": "noting"}
    )
    logger.info(f"Recorded interaction as memory: {memory_id}")

    # 5. Simulate user feedback
    logger.info("User gives positive feedback")
    result = agent.provide_feedback(memory_id, positive=True, reason="User tried it and it helped")
    logger.info(f"   Q-value: {result.old_q:.3f} -> {result.new_q:.3f}")

    # 6. Show stats
    logger.info("Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")

    logger.info("=" * 60)
    logger.info("Integration demo complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    demo_integration()
