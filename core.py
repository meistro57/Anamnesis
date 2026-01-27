"""
Anamnesis - Self-Evolving Memory for AI Agents

ἀνάμνησις (anamnesis) — Greek: "recollection"
Plato's concept that the soul recalls knowledge from past existence.

Based on: "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory"
(Zhang et al., 2026) - arXiv:2601.03192

This implementation provides:
1. Episodic memory storage with Q-value tracking
2. Two-phase retrieval (semantic + utility)
3. Bellman-style Q-value updates from feedback
4. SQLite persistence for testing across sessions

Repository: https://github.com/[your-username]/anamnesis
"""

import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


@dataclass
class EpisodicMemory:
    """A single episode/memory entry"""
    id: str                          # Unique identifier
    task_type: str                   # Category of task (e.g., "code_help", "creative", "research")
    query_summary: str               # Condensed version of user query
    context_summary: str             # Key context elements
    action_taken: str                # What the agent did
    outcome_description: str         # What happened
    success: bool                    # Binary success indicator
    reward: float                    # Numeric reward (-1 to 1)
    q_value: float                   # Learned utility (starts at reward, evolves)
    embedding: Optional[List[float]] # Semantic embedding vector
    retrieval_count: int             # How often this memory was retrieved
    last_retrieved: Optional[str]    # Timestamp of last retrieval
    created_at: str                  # Creation timestamp
    metadata: Dict[str, Any]         # Flexible additional data


class MemRLStore:
    """
    SQLite-backed episodic memory store with Q-value tracking
    """
    
    def __init__(self, db_path: str = "memrl_memories.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                query_summary TEXT NOT NULL,
                context_summary TEXT,
                action_taken TEXT NOT NULL,
                outcome_description TEXT,
                success INTEGER NOT NULL,
                reward REAL NOT NULL,
                q_value REAL NOT NULL,
                embedding TEXT,
                retrieval_count INTEGER DEFAULT 0,
                last_retrieved TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Index for efficient retrieval
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON memories(task_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_value ON memories(q_value DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC)")
        
        # Q-value update history for analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                old_q REAL NOT NULL,
                new_q REAL NOT NULL,
                reward_signal REAL NOT NULL,
                update_reason TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_memory(self, memory: EpisodicMemory) -> str:
        """Store a new episodic memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, task_type, query_summary, context_summary, action_taken, 
             outcome_description, success, reward, q_value, embedding,
             retrieval_count, last_retrieved, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.task_type,
            memory.query_summary,
            memory.context_summary,
            memory.action_taken,
            memory.outcome_description,
            1 if memory.success else 0,
            memory.reward,
            memory.q_value,
            json.dumps(memory.embedding) if memory.embedding else None,
            memory.retrieval_count,
            memory.last_retrieved,
            memory.created_at,
            json.dumps(memory.metadata)
        ))
        
        conn.commit()
        conn.close()
        return memory.id
    
    def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a single memory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_memory(row)
        return None
    
    def _row_to_memory(self, row) -> EpisodicMemory:
        """Convert database row to EpisodicMemory object"""
        return EpisodicMemory(
            id=row[0],
            task_type=row[1],
            query_summary=row[2],
            context_summary=row[3],
            action_taken=row[4],
            outcome_description=row[5],
            success=bool(row[6]),
            reward=row[7],
            q_value=row[8],
            embedding=json.loads(row[9]) if row[9] else None,
            retrieval_count=row[10],
            last_retrieved=row[11],
            created_at=row[12],
            metadata=json.loads(row[13]) if row[13] else {}
        )
    
    def update_q_value(self, memory_id: str, new_q: float, 
                       reward_signal: float, reason: str = "") -> bool:
        """
        Update Q-value for a memory and log the change
        
        This is the core learning mechanism - memories that prove useful
        get higher Q-values, making them more likely to be retrieved.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current Q-value
        cursor.execute("SELECT q_value FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        
        old_q = row[0]
        
        # Update Q-value
        cursor.execute(
            "UPDATE memories SET q_value = ? WHERE id = ?",
            (new_q, memory_id)
        )
        
        # Log the update for analysis
        cursor.execute("""
            INSERT INTO q_updates (memory_id, old_q, new_q, reward_signal, update_reason, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, old_q, new_q, reward_signal, reason, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
    
    def increment_retrieval(self, memory_id: str):
        """Mark a memory as having been retrieved"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET retrieval_count = retrieval_count + 1,
                last_retrieved = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        
        conn.commit()
        conn.close()
    
    def get_all_memories(self, task_type: Optional[str] = None) -> List[EpisodicMemory]:
        """Get all memories, optionally filtered by task type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_type:
            cursor.execute(
                "SELECT * FROM memories WHERE task_type = ? ORDER BY q_value DESC",
                (task_type,)
            )
        else:
            cursor.execute("SELECT * FROM memories ORDER BY q_value DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def get_top_by_q_value(self, n: int = 10, task_type: Optional[str] = None) -> List[EpisodicMemory]:
        """Get top N memories by Q-value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_type:
            cursor.execute(
                "SELECT * FROM memories WHERE task_type = ? ORDER BY q_value DESC LIMIT ?",
                (task_type, n)
            )
        else:
            cursor.execute("SELECT * FROM memories ORDER BY q_value DESC LIMIT ?", (n,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def get_q_update_history(self, memory_id: str) -> List[Dict]:
        """Get the Q-value update history for a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT old_q, new_q, reward_signal, update_reason, updated_at
            FROM q_updates WHERE memory_id = ? ORDER BY updated_at
        """, (memory_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "old_q": row[0],
                "new_q": row[1],
                "reward_signal": row[2],
                "reason": row[3],
                "timestamp": row[4]
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the memory store"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        stats["total_memories"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(q_value) FROM memories")
        stats["avg_q_value"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT task_type, COUNT(*) FROM memories GROUP BY task_type")
        stats["by_task_type"] = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM memories WHERE success = 1")
        stats["successful_memories"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(retrieval_count) FROM memories")
        stats["total_retrievals"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM q_updates")
        stats["total_q_updates"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


class TwoPhaseRetriever:
    """
    Implements MemRL's Two-Phase Retrieval:
    
    Phase A: Semantic filtering - find candidates similar to query
    Phase B: Utility ranking - rank by Q-value
    """
    
    def __init__(self, store: MemRLStore, embedding_fn=None):
        """
        Args:
            store: MemRLStore instance
            embedding_fn: Function that takes text and returns embedding vector
                         If None, uses simple keyword matching
        """
        self.store = store
        self.embedding_fn = embedding_fn
    
    def retrieve(self, 
                 query: str,
                 task_type: Optional[str] = None,
                 phase_a_k: int = 20,      # Candidates from semantic search
                 phase_b_n: int = 5,       # Final results after Q-ranking
                 min_q_threshold: float = -0.5  # Minimum Q-value to consider
                 ) -> List[Tuple[EpisodicMemory, float]]:
        """
        Two-phase retrieval:
        1. Get top-K semantically similar memories
        2. Rank by Q-value, return top-N
        
        Returns: List of (memory, similarity_score) tuples
        """
        
        # Phase A: Semantic filtering
        candidates = self._phase_a_semantic_filter(query, task_type, phase_a_k)
        
        # Phase B: Q-value ranking
        results = self._phase_b_utility_ranking(candidates, phase_b_n, min_q_threshold)
        
        # Mark as retrieved and return
        for memory, _ in results:
            self.store.increment_retrieval(memory.id)
        
        return results
    
    def _phase_a_semantic_filter(self, 
                                  query: str, 
                                  task_type: Optional[str],
                                  k: int) -> List[Tuple[EpisodicMemory, float]]:
        """
        Phase A: Find K most semantically similar memories
        """
        memories = self.store.get_all_memories(task_type)
        
        if not memories:
            return []
        
        if self.embedding_fn and memories[0].embedding:
            # Use embedding similarity
            query_embedding = self.embedding_fn(query)
            scored = []
            
            for mem in memories:
                if mem.embedding:
                    sim = self._cosine_similarity(query_embedding, mem.embedding)
                    scored.append((mem, sim))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]
        else:
            # Fallback: keyword matching
            query_words = set(query.lower().split())
            scored = []
            
            for mem in memories:
                mem_text = f"{mem.query_summary} {mem.action_taken} {mem.context_summary}".lower()
                mem_words = set(mem_text.split())
                
                overlap = len(query_words & mem_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), 1)
                    scored.append((mem, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]
    
    def _phase_b_utility_ranking(self,
                                  candidates: List[Tuple[EpisodicMemory, float]],
                                  n: int,
                                  min_q: float) -> List[Tuple[EpisodicMemory, float]]:
        """
        Phase B: Rank candidates by Q-value (utility)
        
        The key insight: semantic similarity gets us relevant memories,
        but Q-value tells us which relevant memories are actually *useful*
        """
        # Filter by minimum Q-value
        filtered = [(mem, sim) for mem, sim in candidates if mem.q_value >= min_q]
        
        # Sort by Q-value (utility), breaking ties with similarity
        filtered.sort(key=lambda x: (x[0].q_value, x[1]), reverse=True)
        
        return filtered[:n]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class QLearner:
    """
    Implements Q-value learning from feedback
    
    Uses a simplified Bellman update:
    Q(s) = Q(s) + α * (reward - Q(s))
    
    Where reward comes from:
    - Explicit user feedback (thumbs up/down)
    - Task completion signals
    - Implicit signals (was the retrieved memory actually used?)
    """
    
    def __init__(self, store: MemRLStore, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9):
        self.store = store
        self.alpha = learning_rate
        self.gamma = discount_factor
    
    def update_from_feedback(self, 
                             memory_id: str, 
                             reward: float,
                             reason: str = "") -> float:
        """
        Update Q-value based on feedback
        
        Args:
            memory_id: ID of memory that was used
            reward: Reward signal (-1 to 1)
            reason: Description of why this update is happening
        
        Returns: New Q-value
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            return 0.0
        
        # Bellman update (simplified, no next state)
        new_q = memory.q_value + self.alpha * (reward - memory.q_value)
        
        # Clamp to reasonable range
        new_q = max(-1.0, min(1.0, new_q))
        
        self.store.update_q_value(memory_id, new_q, reward, reason)
        
        return new_q
    
    def batch_update(self, 
                     used_memories: List[str],
                     task_success: bool,
                     explicit_feedback: Optional[float] = None) -> Dict[str, float]:
        """
        Update Q-values for all memories used in a task
        
        Args:
            used_memories: List of memory IDs that were retrieved/used
            task_success: Whether the overall task succeeded
            explicit_feedback: Optional user rating (-1 to 1)
        
        Returns: Dict mapping memory_id to new Q-value
        """
        # Determine reward signal
        if explicit_feedback is not None:
            base_reward = explicit_feedback
        else:
            base_reward = 0.5 if task_success else -0.3
        
        updates = {}
        
        for i, mem_id in enumerate(used_memories):
            # Earlier memories in the chain get slightly less credit
            # (recency weighting)
            position_weight = 0.8 ** i
            reward = base_reward * position_weight
            
            reason = f"Task {'succeeded' if task_success else 'failed'}"
            if explicit_feedback is not None:
                reason += f" (user feedback: {explicit_feedback})"
            
            new_q = self.update_from_feedback(mem_id, reward, reason)
            updates[mem_id] = new_q
        
        return updates
    
    def decay_unused(self, decay_factor: float = 0.99, 
                     days_unused: int = 30) -> int:
        """
        Apply decay to memories that haven't been retrieved recently
        
        This prevents old, unused memories from dominating just because
        they had early success.
        
        Returns: Number of memories decayed
        """
        from datetime import datetime, timedelta
        
        cutoff = (datetime.now() - timedelta(days=days_unused)).isoformat()
        
        conn = sqlite3.connect(self.store.db_path)
        cursor = conn.cursor()
        
        # Find memories not retrieved recently
        cursor.execute("""
            SELECT id, q_value FROM memories 
            WHERE (last_retrieved IS NULL OR last_retrieved < ?)
            AND q_value > 0
        """, (cutoff,))
        
        rows = cursor.fetchall()
        count = 0
        
        for mem_id, current_q in rows:
            new_q = current_q * decay_factor
            self.store.update_q_value(mem_id, new_q, 0, f"Decay (unused {days_unused}+ days)")
            count += 1
        
        conn.close()
        return count


def create_memory_id(query: str, action: str) -> str:
    """Generate a unique ID for a memory"""
    content = f"{query}|{action}|{datetime.now().isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================
# Example Usage / Test Harness
# ============================================

def demo():
    """Demonstrate the MemRL-Lite system"""
    
    print("=" * 60)
    print("MemRL-Lite Demo")
    print("=" * 60)
    
    # Initialize
    store = MemRLStore("demo_memories.db")
    retriever = TwoPhaseRetriever(store)
    learner = QLearner(store, learning_rate=0.2)
    
    # Create some sample memories
    sample_memories = [
        EpisodicMemory(
            id=create_memory_id("python loop", "provided for loop example"),
            task_type="code_help",
            query_summary="How to write a for loop in Python",
            context_summary="User learning Python basics",
            action_taken="Provided for loop syntax with enumerate example",
            outcome_description="User successfully wrote their loop",
            success=True,
            reward=0.8,
            q_value=0.8,
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=datetime.now().isoformat(),
            metadata={"language": "python", "level": "beginner"}
        ),
        EpisodicMemory(
            id=create_memory_id("python loop", "showed list comprehension"),
            task_type="code_help",
            query_summary="How to iterate through a list",
            context_summary="User working on data processing",
            action_taken="Showed list comprehension instead of loop",
            outcome_description="User found it confusing, asked for simpler approach",
            success=False,
            reward=-0.3,
            q_value=-0.3,
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=datetime.now().isoformat(),
            metadata={"language": "python", "level": "beginner"}
        ),
        EpisodicMemory(
            id=create_memory_id("consciousness", "seth speaks reference"),
            task_type="spiritual_guidance",
            query_summary="Understanding reality creation",
            context_summary="User exploring Seth material",
            action_taken="Referenced Seth's belief-creates-reality framework",
            outcome_description="User had breakthrough insight",
            success=True,
            reward=0.9,
            q_value=0.9,
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=datetime.now().isoformat(),
            metadata={"tradition": "seth", "concept": "reality_creation"}
        ),
    ]
    
    print("\n📝 Storing sample memories...")
    for mem in sample_memories:
        store.store_memory(mem)
        print(f"  - Stored: {mem.query_summary[:40]}... (Q={mem.q_value})")
    
    # Test retrieval
    print("\n🔍 Testing Two-Phase Retrieval...")
    
    query = "How do I loop through items in Python?"
    print(f"\nQuery: '{query}'")
    
    results = retriever.retrieve(query, task_type="code_help", phase_b_n=3)
    
    print(f"Results (ranked by Q-value):")
    for mem, sim in results:
        print(f"  - Q={mem.q_value:.2f} | Sim={sim:.2f} | {mem.action_taken[:50]}")
    
    # Simulate feedback
    if results:
        print("\n📊 Simulating user feedback...")
        best_match = results[0][0]
        print(f"User gave positive feedback on: {best_match.action_taken[:40]}...")
        
        old_q = best_match.q_value
        new_q = learner.update_from_feedback(best_match.id, reward=1.0, reason="User explicit thumbs up")
        print(f"Q-value updated: {old_q:.2f} → {new_q:.2f}")
    
    # Show stats
    print("\n📈 Memory Store Statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
