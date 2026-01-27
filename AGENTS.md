# Anamnesis Agent Development Guide

## 🌟 Project Overview

Anamnesis is a self-evolving memory system for AI agents that goes beyond traditional RAG (Retrieval-Augmented Generation) by incorporating Q-value learning to rank memories based on their utility.

## 🔧 Project Setup

### Python Requirements
- Python 3.8 to 3.12
- Required dependencies: 
  - numpy (>=1.20.0)
- Optional dependencies:
  - OpenAI embeddings
  - Sentence Transformers

### Development Dependencies
- pytest (>=7.0.0)
- pytest-cov (>=4.0.0)
- black (>=23.0.0)
- ruff (>=0.1.0)

## 📋 Code Style and Conventions

### Formatting
- Black formatter
- Line length: 100 characters
- Target Python versions: 3.8 through 3.12

### Linting
- Using ruff
- Enabled checks: E, F, W, I, N
- Ignore E501 (line length, let Black handle it)

## 🧪 Testing

### Test Command
```bash
python test_harness.py
```

### Test Characteristics
- Located in `test_harness.py`
- Simulates 100 user interactions
- Validates Q-value learning mechanism
- Checks memory ranking based on success rates

### Expected Test Output
- Shows average Q-values for successful and failed memories
- Verifies that successful memories have higher Q-values

## 🤖 Key Components

### Core Files
- `core.py`: Storage, retrieval, Q-learning implementation
- `agent.py`: High-level API for agent integration
- `test_harness.py`: Test suite for Q-value learning

## 🚀 Development Tips

### Memory Management
- Use task types to categorize memories
- Provide feedback to help memories learn
- Can use positive/negative or fine-grained (-1 to 1) ratings

### Integration Patterns
- Initialize with a database path
- Get context before generating responses
- Record interactions
- Provide feedback to improve memory utility

## 🔍 Key Concepts

### Q-Value Learning
- Learning rate: Default 0.15
- Reward range: -1 to 1
- Q-value update formula:
  ```
  Q(memory) = Q(memory) + α * (reward - Q(memory))
  ```

### Memory Decay
- Unused memories decay after 30 days
- Prevents stale memories from dominating

## 🚧 Limitations and Considerations

- Current version supports single-user scenarios
- Embeddings recommended for better semantic search
- Memory bank per application/use case

## 🔮 Future Development Areas
- Multi-user support
- Memory consolidation
- Hierarchical memory structures
- Active forgetting mechanism
- Transfer learning across domains

## 📚 References
- [Original MemRL Paper](https://arxiv.org/abs/2601.03192)
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

---

*Built for exploring the frontiers of AI memory and learning.*