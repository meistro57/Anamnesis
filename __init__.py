"""
Anamnesis - Self-Evolving Memory for AI Agents

ἀνάμνησις (anamnesis) — Greek: "recollection"
Plato's concept that the soul recalls knowledge from past existence.

Usage:
    from anamnesis import AnamnesisAgent
    
    agent = AnamnesisAgent("./memories.db")
    
    # Retrieve relevant context
    context = agent.get_context("user query", task_type="code_help")
    
    # Record interactions
    memory_id = agent.record_interaction(
        task_type="code_help",
        query="user query",
        response="assistant response",
        success=True
    )
    
    # Learn from feedback
    agent.provide_feedback(memory_id, positive=True)
"""

from .core import (
    EpisodicMemory,
    MemRLStore,
    TwoPhaseRetriever,
    QLearner,
    create_memory_id,
)

from .agent import (
    AnamnesisAgent,
    RetrievalResult,
    FeedbackResult,
    create_openai_embedding_fn,
    create_sentence_transformer_fn,
)

__version__ = "0.1.0"
__author__ = "Quantum Minds United"
__all__ = [
    "AnamnesisAgent",
    "EpisodicMemory",
    "RetrievalResult",
    "FeedbackResult",
    "MemRLStore",
    "TwoPhaseRetriever",
    "QLearner",
    "create_memory_id",
    "create_openai_embedding_fn",
    "create_sentence_transformer_fn",
]
