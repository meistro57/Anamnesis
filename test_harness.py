"""
Anamnesis Test Harness

This script simulates realistic usage patterns to demonstrate
that the Q-value learning actually works:

1. Creates memories with varied initial success rates
2. Simulates retrieval patterns
3. Provides feedback that should cause Q-values to evolve
4. Verifies that high-utility memories rise to the top

The key test: After learning, the retriever should prefer
memories that actually helped, not just ones that are similar.
"""

import os
import sys
import random
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    MemRLStore, TwoPhaseRetriever, QLearner, 
    EpisodicMemory, create_memory_id
)


def generate_test_memories():
    """
    Generate a diverse set of test memories with realistic patterns
    
    Scenario: A coding assistant with memories of different approaches
    Some approaches work well, others don't - but they're all semantically similar
    """
    memories = []
    
    # Python loop questions - different approaches with different success rates
    loop_scenarios = [
        {
            "query": "iterate through list python",
            "action": "Used simple for loop with index",
            "success": True,
            "reward": 0.7,
            "note": "Basic but reliable"
        },
        {
            "query": "loop through list python",
            "action": "Used enumerate() for index and value",
            "success": True,
            "reward": 0.9,
            "note": "Pythonic, well-received"
        },
        {
            "query": "python list iteration",
            "action": "Used list comprehension",
            "success": False,
            "reward": -0.4,
            "note": "Too advanced for beginner"
        },
        {
            "query": "for loop python list",
            "action": "Used while loop with counter",
            "success": False,
            "reward": -0.2,
            "note": "Worked but not idiomatic"
        },
        {
            "query": "iterate python array",
            "action": "Used map() function",
            "success": False,
            "reward": -0.5,
            "note": "Confused the user"
        },
    ]
    
    # Error handling questions
    error_scenarios = [
        {
            "query": "python try except",
            "action": "Showed basic try/except with specific exception",
            "success": True,
            "reward": 0.8,
            "note": "Clear and practical"
        },
        {
            "query": "handle errors python",
            "action": "Explained exception hierarchy",
            "success": False,
            "reward": -0.3,
            "note": "Too theoretical"
        },
        {
            "query": "python error handling",
            "action": "Provided try/except/finally template",
            "success": True,
            "reward": 0.6,
            "note": "Useful reference"
        },
    ]
    
    # Spiritual/consciousness questions (for Eli GPT / Awakening Mind context)
    consciousness_scenarios = [
        {
            "query": "reality creation beliefs",
            "action": "Referenced Seth's point of power in present",
            "success": True,
            "reward": 0.9,
            "note": "Resonated deeply"
        },
        {
            "query": "manifesting reality",
            "action": "Gave generic law of attraction advice",
            "success": False,
            "reward": -0.2,
            "note": "Too surface level"
        },
        {
            "query": "creating your reality",
            "action": "Connected to user's specific situation with Seth framework",
            "success": True,
            "reward": 0.95,
            "note": "Personalized breakthrough"
        },
        {
            "query": "beliefs shape reality",
            "action": "Explained limiting beliefs concept",
            "success": True,
            "reward": 0.7,
            "note": "Good foundation"
        },
    ]
    
    # Create memory objects
    base_time = datetime.now() - timedelta(days=30)
    
    for i, scenario in enumerate(loop_scenarios):
        mem = EpisodicMemory(
            id=create_memory_id(scenario["query"], scenario["action"]),
            task_type="code_help",
            query_summary=scenario["query"],
            context_summary="User learning Python basics",
            action_taken=scenario["action"],
            outcome_description=scenario["note"],
            success=scenario["success"],
            reward=scenario["reward"],
            q_value=scenario["reward"],  # Initialize Q to reward
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=(base_time + timedelta(days=i)).isoformat(),
            metadata={"domain": "python", "topic": "loops"}
        )
        memories.append(mem)
    
    for i, scenario in enumerate(error_scenarios):
        mem = EpisodicMemory(
            id=create_memory_id(scenario["query"], scenario["action"]),
            task_type="code_help",
            query_summary=scenario["query"],
            context_summary="User handling errors",
            action_taken=scenario["action"],
            outcome_description=scenario["note"],
            success=scenario["success"],
            reward=scenario["reward"],
            q_value=scenario["reward"],
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=(base_time + timedelta(days=i+5)).isoformat(),
            metadata={"domain": "python", "topic": "errors"}
        )
        memories.append(mem)
    
    for i, scenario in enumerate(consciousness_scenarios):
        mem = EpisodicMemory(
            id=create_memory_id(scenario["query"], scenario["action"]),
            task_type="spiritual_guidance",
            query_summary=scenario["query"],
            context_summary="User exploring consciousness",
            action_taken=scenario["action"],
            outcome_description=scenario["note"],
            success=scenario["success"],
            reward=scenario["reward"],
            q_value=scenario["reward"],
            embedding=None,
            retrieval_count=0,
            last_retrieved=None,
            created_at=(base_time + timedelta(days=i+10)).isoformat(),
            metadata={"domain": "consciousness", "tradition": "seth"}
        )
        memories.append(mem)
    
    return memories


def simulate_usage(store: MemRLStore, retriever: TwoPhaseRetriever, 
                   learner: QLearner, num_interactions: int = 50):
    """
    Simulate realistic usage patterns:
    - User asks questions
    - System retrieves relevant memories
    - User provides feedback
    - Q-values evolve
    """
    
    test_queries = [
        ("How do I loop through a list in Python?", "code_help"),
        ("Python iterate array", "code_help"),
        ("for loop python", "code_help"),
        ("handle exceptions python", "code_help"),
        ("python error catching", "code_help"),
        ("How do beliefs create reality?", "spiritual_guidance"),
        ("Seth material reality creation", "spiritual_guidance"),
        ("manifesting with beliefs", "spiritual_guidance"),
    ]
    
    print("\n🔄 Simulating usage patterns...")
    print("-" * 50)
    
    for i in range(num_interactions):
        # Pick a random query
        query, task_type = random.choice(test_queries)
        
        # Retrieve relevant memories
        results = retriever.retrieve(query, task_type=task_type, phase_b_n=3)
        
        if not results:
            continue
        
        # Simulate user feedback based on the memory that was surfaced
        top_memory, similarity = results[0]
        
        # Successful memories get positive feedback more often
        # This simulates: good advice → happy user → positive feedback
        if top_memory.success:
            feedback = random.choice([0.5, 0.7, 0.9, 1.0])
        else:
            feedback = random.choice([-0.5, -0.3, 0.0, 0.2])
        
        # Update Q-value
        learner.update_from_feedback(
            top_memory.id, 
            feedback,
            f"Interaction {i+1}: User query '{query[:30]}...'"
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_interactions} interactions")
    
    print("\n✅ Simulation complete!")


def verify_learning(store: MemRLStore, retriever: TwoPhaseRetriever):
    """
    Verify that Q-value learning worked:
    - High-utility memories should have risen
    - Low-utility memories should have fallen
    - Retrieval should now prefer successful approaches
    """
    print("\n" + "=" * 60)
    print("📊 VERIFICATION: Did the learning work?")
    print("=" * 60)
    
    # Check Q-value distribution
    all_memories = store.get_all_memories()
    
    print("\n1️⃣ Q-Value Distribution After Learning:")
    print("-" * 40)
    
    successful = [m for m in all_memories if m.success]
    failed = [m for m in all_memories if not m.success]
    
    avg_success_q = sum(m.q_value for m in successful) / len(successful) if successful else 0
    avg_fail_q = sum(m.q_value for m in failed) / len(failed) if failed else 0
    
    print(f"  Average Q-value for SUCCESSFUL memories: {avg_success_q:.3f}")
    print(f"  Average Q-value for FAILED memories:     {avg_fail_q:.3f}")
    print(f"  Difference (should be positive):         {avg_success_q - avg_fail_q:.3f}")
    
    if avg_success_q > avg_fail_q:
        print("  ✅ SUCCESS: Good memories have higher Q-values!")
    else:
        print("  ❌ ISSUE: Learning may not be working correctly")
    
    # Test retrieval behavior
    print("\n2️⃣ Retrieval Behavior Test:")
    print("-" * 40)
    
    test_cases = [
        ("loop through list python", "code_help"),
        ("reality creation beliefs", "spiritual_guidance"),
    ]
    
    for query, task_type in test_cases:
        print(f"\n  Query: '{query}'")
        results = retriever.retrieve(query, task_type=task_type, phase_b_n=3)
        
        if results:
            print("  Top results (ranked by Q-value):")
            for mem, sim in results:
                status = "✅" if mem.success else "❌"
                print(f"    {status} Q={mem.q_value:.2f} | {mem.action_taken[:45]}...")
            
            # Check if top result is successful
            if results[0][0].success:
                print("  ✅ Top result is a SUCCESSFUL memory!")
            else:
                print("  ⚠️  Top result is a FAILED memory (might need more learning)")
    
    # Show top memories overall
    print("\n3️⃣ Top 5 Memories by Q-Value (Overall):")
    print("-" * 40)
    
    top_memories = store.get_top_by_q_value(5)
    for i, mem in enumerate(top_memories):
        status = "✅" if mem.success else "❌"
        print(f"  {i+1}. {status} Q={mem.q_value:.3f} | {mem.action_taken[:50]}...")
    
    # Show Q-value evolution for one memory
    print("\n4️⃣ Q-Value Evolution Example:")
    print("-" * 40)
    
    if top_memories:
        sample_mem = top_memories[0]
        history = store.get_q_update_history(sample_mem.id)
        
        print(f"  Memory: {sample_mem.action_taken[:50]}...")
        print(f"  Initial Q: {sample_mem.reward:.3f}")
        print(f"  Updates: {len(history)}")
        
        if history:
            print("  Evolution:")
            for h in history[:5]:  # Show first 5 updates
                print(f"    {h['old_q']:.3f} → {h['new_q']:.3f} (reward: {h['reward_signal']:.2f})")
            if len(history) > 5:
                print(f"    ... and {len(history) - 5} more updates")
    
    print("\n" + "=" * 60)


def run_full_test():
    """Run the complete test suite"""
    
    print("\n" + "=" * 60)
    print("Anamnesis Test Suite")
    print("Testing Q-value learning for self-evolving memory")
    print("=" * 60)
    
    # Clean start
    db_path = "test_memrl.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Initialize components
    store = MemRLStore(db_path)
    retriever = TwoPhaseRetriever(store)
    learner = QLearner(store, learning_rate=0.15)
    
    # Step 1: Generate and store test memories
    print("\n📝 Step 1: Generating test memories...")
    memories = generate_test_memories()
    for mem in memories:
        store.store_memory(mem)
    print(f"  Stored {len(memories)} memories")
    
    # Step 2: Show initial state
    print("\n📊 Step 2: Initial Q-value state...")
    initial_stats = store.get_stats()
    print(f"  Total memories: {initial_stats['total_memories']}")
    print(f"  Average Q-value: {initial_stats['avg_q_value']:.3f}")
    
    # Step 3: Simulate usage
    print("\n🔄 Step 3: Simulating 100 user interactions...")
    simulate_usage(store, retriever, learner, num_interactions=100)
    
    # Step 4: Verify learning
    print("\n✅ Step 4: Verifying learning worked...")
    verify_learning(store, retriever)
    
    # Final stats
    print("\n📈 Final Statistics:")
    print("-" * 40)
    final_stats = store.get_stats()
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  Average Q-value: {final_stats['avg_q_value']:.3f}")
    print(f"  Total retrievals: {final_stats['total_retrievals']}")
    print(f"  Total Q-updates: {final_stats['total_q_updates']}")
    
    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)
    
    return store, retriever, learner


if __name__ == "__main__":
    store, retriever, learner = run_full_test()
