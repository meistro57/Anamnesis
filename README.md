# Anamnesis

**Self-Evolving Memory for AI Agents**

> *ἀνάμνησις (anamnesis)* — Greek: "recollection" — Plato's concept that the soul recalls knowledge from past existence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Based on: [MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory](https://arxiv.org/abs/2601.03192) (Zhang et al., 2026)

---

## 🧠 The Big Idea

Traditional AI memory systems (like RAG) retrieve context based on **semantic similarity** — "what's similar to the current query?" But similarity ≠ usefulness. 

Anamnesis adds a crucial second dimension: **utility** (Q-values learned from feedback).

```
Traditional RAG:  Query → Semantic Search → Results
Anamnesis:        Query → Semantic Search → Q-Value Ranking → Results
                                                 ↑
                                          (learned from feedback)
```

**The key insight:** Memories that *actually helped* users should rank higher than memories that just *seem relevant*.

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Anamnesis                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────────────┐                  │
│   │   Query     │────▶│  Phase A: Semantic  │                  │
│   │             │     │  Filter (top-K)     │                  │
│   └─────────────┘     └──────────┬──────────┘                  │
│                                  │                              │
│                                  ▼                              │
│                       ┌─────────────────────┐                  │
│                       │  Phase B: Q-Value   │                  │
│                       │  Ranking (top-N)    │                  │
│                       └──────────┬──────────┘                  │
│                                  │                              │
│                                  ▼                              │
│   ┌─────────────┐     ┌─────────────────────┐                  │
│   │  Response   │◀────│  Inject as Context  │                  │
│   │  Generation │     └─────────────────────┘                  │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐     ┌─────────────────────┐                  │
│   │   Feedback  │────▶│  Q-Value Update     │                  │
│   │  (👍 / 👎)  │     │  (Bellman Update)   │                  │
│   └─────────────┘     └──────────┬──────────┘                  │
│                                  │                              │
│                                  ▼                              │
│                       ┌─────────────────────┐                  │
│                       │   Episodic Memory   │                  │
│                       │   Store (SQLite)    │                  │
│                       │                     │                  │
│                       │  ┌───────────────┐  │                  │
│                       │  │ query_summary │  │                  │
│                       │  │ action_taken  │  │                  │
│                       │  │ success       │  │                  │
│                       │  │ Q_value ◀─────│──│── Updated!       │
│                       │  │ embedding     │  │                  │
│                       │  └───────────────┘  │                  │
│                       └─────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from anamnesis import AnamnesisAgent

# Initialize
agent = AnamnesisAgent("./memories.db")

# When responding to a user query
query = "How do I create a for loop in Python?"

# 1. Get relevant context from past experiences
context = agent.get_context(query, task_type="code_help")

# 2. Format for your LLM prompt
prompt_context = agent.format_context_for_prompt(context)
# This gives you XML-formatted past experiences to inject

# 3. After your LLM responds, record the interaction
memory_id = agent.record_interaction(
    task_type="code_help",
    query=query,
    response="Use 'for item in list:' syntax...",
    success=True
)

# 4. When user gives feedback
agent.provide_feedback(memory_id, positive=True)  # 👍
# or
agent.provide_feedback(memory_id, positive=False)  # 👎
```

### With Embeddings (Better Semantic Search)

```python
from anamnesis import AnamnesisAgent, create_openai_embedding_fn

# Using OpenAI embeddings
embedding_fn = create_openai_embedding_fn(api_key="sk-...")

agent = AnamnesisAgent(
    db_path="./memories.db",
    embedding_fn=embedding_fn
)

# Or using local embeddings (no API needed)
from anamnesis import create_sentence_transformer_fn

embedding_fn = create_sentence_transformer_fn("all-MiniLM-L6-v2")
agent = AnamnesisAgent(embedding_fn=embedding_fn)
```

## 📁 Files

| File | Description |
|------|-------------|
| `core.py` | Core implementation: storage, retrieval, Q-learning |
| `agent.py` | High-level API for integration |
| `test_harness.py` | Test suite validating Q-value learning works |

## 🧪 Running Tests

```bash
# Run the full test suite
python test_harness.py
```

This will:
1. Create test memories with varied success rates
2. Simulate 100 user interactions
3. Verify that Q-values evolve correctly
4. Show that successful memories rise to the top

Expected output:
```
Average Q-value for SUCCESSFUL memories: 0.731
Average Q-value for FAILED memories:     -0.320
Difference (should be positive):         1.051
✅ SUCCESS: Good memories have higher Q-values!
```

## 🔧 Integration Points

### Chat Bridge Integration

```python
# In your Chat Bridge message handler

from anamnesis import AnamnesisAgent

agent = AnamnesisAgent("./chat_bridge_memories.db")

async def handle_message(user_message, conversation_id):
    # Get context from past similar conversations
    context = agent.get_context(
        user_message,
        task_type=infer_task_type(user_message)
    )
    
    # Add to your prompt
    system_prompt = f"""
    {your_base_prompt}
    
    {agent.format_context_for_prompt(context)}
    """
    
    # Get LLM response
    response = await get_llm_response(system_prompt, user_message)
    
    # Record for future learning
    memory_id = agent.record_interaction(
        task_type=infer_task_type(user_message),
        query=user_message,
        response=response,
        metadata={"conversation_id": conversation_id}
    )
    
    return response, memory_id
```

### Eli GPT / Awakening Mind GPT

```python
# Specialized for consciousness/spiritual domain

from anamnesis import AnamnesisAgent

agent = AnamnesisAgent("./eli_memories.db")

# Task types you might use:
# - "reality_creation" - Seth-based manifestation
# - "meditation" - Practical guidance
# - "awakening" - Consciousness exploration
# - "shadow_work" - Integration practices

# Example: Track which Seth references resonate
memory_id = agent.record_interaction(
    task_type="reality_creation",
    query="How do beliefs create reality?",
    response="As Seth teaches, the point of power is in the present...",
    context="User exploring Seth Speaks material",
    success=True,
    metadata={
        "tradition": "seth",
        "concept": "point_of_power",
        "book_reference": "Seth Speaks Ch. 15"
    }
)
```

### Feedback Mechanisms

```python
# Simple thumbs up/down
agent.provide_feedback(memory_id, positive=True)

# Fine-grained rating
agent.provide_feedback(memory_id, rating=0.8)  # -1 to 1

# Bulk feedback for all memories used in conversation
agent.bulk_feedback(task_success=True, explicit_rating=0.9)

# Implicit feedback (memory was retrieved but not useful)
# Just don't call provide_feedback - Q-values decay naturally
agent.run_maintenance(decay_unused_days=30)
```

## 📈 Q-Value Learning Details

### The Bellman Update

```python
Q(memory) = Q(memory) + α * (reward - Q(memory))
```

Where:
- `α` = learning rate (default 0.15)
- `reward` = feedback signal (-1 to 1)

### What Affects Q-Values

| Event | Effect |
|-------|--------|
| 👍 Positive feedback | Q increases (+0.8 typical) |
| 👎 Negative feedback | Q decreases (-0.5 typical) |
| Task success | Q increases for all used memories |
| Task failure | Q decreases for all used memories |
| Memory unused for 30+ days | Q slowly decays (prevents stale dominance) |

### Observing Q-Value Evolution

```python
# Get update history for a memory
history = store.get_q_update_history(memory_id)

for h in history:
    print(f"{h['old_q']:.3f} → {h['new_q']:.3f} ({h['reason']})")
```

## 🎯 Use Cases for Your Projects

### Quantum Minds United
- Eli GPT learns which consciousness frameworks resonate with different users
- Track which meditation techniques actually help people
- Remember breakthrough insights to offer similar guidance

### Podcast Visual Factory
- Learn which visual styles get positive engagement
- Remember successful prompt patterns for different content types

## ⚙️ Configuration Options

```python
agent = AnamnesisAgent(
    db_path="./memories.db",
    
    # Embedding function (optional, improves semantic search)
    embedding_fn=your_embedding_function,
    
    # How fast Q-values adapt (higher = faster but less stable)
    learning_rate=0.15,
    
    # Callbacks for integration
    on_memory_stored=lambda mem: log_memory(mem),
    on_feedback=lambda id, old, new: log_learning(id, old, new)
)
```

## 🔮 Future Enhancements

1. **Multi-user support** - Separate memory banks per user
2. **Memory consolidation** - Merge similar successful memories
3. **Hierarchical memory** - Episode → Session → Long-term
4. **Active forgetting** - Remove consistently unhelpful memories
5. **Transfer learning** - Bootstrap from similar domains

## 📚 References

- [MemRL Paper](https://arxiv.org/abs/2601.03192) - The original research
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) - Comprehensive survey
- [EM-LLM](https://github.com/em-llm/EM-LLM-model) - Related episodic memory work

---

*"The soul is immortal, and has been born many times, and has beheld all things both in this world and in the nether realms... learning is nothing but recollection."* — Plato, Meno

Built for [Quantum Minds United](https://quantummindsunited.com) 🌌
