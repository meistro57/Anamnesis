import os
import pytest
from sentence_transformers import SentenceTransformer
import ollama

from anamnesis import AnamnesisAgent

class TestOllamaIntegration:
    def setup_method(self):
        """
        Set up test environment for Ollama-based Anamnesis testing
        """
        # Ensure Ollama is running and a suitable model is available
        self.test_db_path = "/tmp/ollama_anamnesis_test.db"
        
        # Check Ollama availability
        try:
            ollama_models = ollama.list()
            available_models = [model['name'] for model in ollama_models['models']]
            
            # Preferred models in order of preference
            preferred_models = [
                "mistral:latest", 
                "llama2:13b", 
                "phi:latest", 
                "zephyr:latest"
            ]
            
            self.llm_model = next((model for model in preferred_models if model in available_models), None)
            
            if not self.llm_model:
                raise ValueError("No suitable Ollama model found. Please pull at least one model.")
        
        except Exception as e:
            pytest.fail(f"Ollama setup failed: {e}")
        
        # Create local embedding function
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        def local_embedding_fn(text):
            return self.embedding_model.encode(text).tolist()
        
        # Custom Ollama inference function
        def ollama_inference(prompt):
            response = ollama.chat(model=self.llm_model, messages=[
                {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        
        # Initialize Anamnesis Agent
        self.agent = AnamnesisAgent(
            db_path=self.test_db_path,
            embedding_fn=local_embedding_fn,
            inference_fn=ollama_inference
        )
    
    def test_memory_recording(self):
        """
        Test basic memory recording and retrieval with Ollama
        """
        # Record multiple memories with different task types
        memories = [
            {
                "task_type": "coding",
                "query": "How to write a Python list comprehension?",
                "response": "Use square brackets with an expression: [x for x in iterable if condition]",
                "success": True
            },
            {
                "task_type": "writing",
                "query": "Tips for writing a good introduction",
                "response": "Start with a hook, provide context, state your thesis clearly.",
                "success": True
            },
            {
                "task_type": "debugging",
                "query": "Common Python runtime errors",
                "response": "KeyError, TypeError, ValueError are frequent runtime exceptions.",
                "success": False
            }
        ]
        
        memory_ids = []
        for mem in memories:
            memory_id = self.agent.record_interaction(
                task_type=mem['task_type'],
                query=mem['query'],
                response=mem['response'],
                success=mem['success']
            )
            memory_ids.append(memory_id)
            
            # Provide feedback
            self.agent.provide_feedback(memory_id, positive=mem['success'])
        
        # Retrieve context for a new query
        new_query = "Python list comprehension syntax"
        context = self.agent.get_context(new_query, task_type="coding", top_k=3)
        
        assert len(context) > 0, "Context retrieval failed"
        assert any("list comprehension" in item.get('query', '').lower() for item in context), "Relevant memory not retrieved"
    
    def test_embedding_similarity(self):
        """
        Test embedding-based memory retrieval
        """
        # Record memories with semantic nuances
        test_memories = [
            "Python list comprehension",
            "List manipulation in Python",
            "Advanced Python data structures",
            "Machine learning list operations"
        ]
        
        for memory in test_memories:
            self.agent.record_interaction(
                task_type="coding",
                query=memory,
                response=f"Details about: {memory}",
                success=True
            )
        
        # Test retrieval with semantically similar query
        context = self.agent.get_context("Python list methods", task_type="coding", top_k=2)
        
        assert len(context) > 0, "Semantic retrieval failed"
        
    def test_q_value_learning(self):
        """
        Validate Q-value learning mechanism
        """
        # Record memories with mixed success
        memories = [
            {"query": "Python list methods", "success": True},
            {"query": "Debugging network issues", "success": False},
            {"query": "Machine learning algorithms", "success": True}
        ]
        
        memory_ids = []
        for mem in memories:
            memory_id = self.agent.record_interaction(
                task_type="coding",
                query=mem['query'],
                response=f"Response to {mem['query']}",
                success=mem['success']
            )
            memory_ids.append(memory_id)
            
            # Provide feedback
            self.agent.provide_feedback(memory_id, positive=mem['success'])
        
        # Fetch Q-values and verify learning
        q_values = [self.agent.get_memory_q_value(mid) for mid in memory_ids]
        
        successful_memories = [qv for qv, mem in zip(q_values, memories) if mem['success']]
        failed_memories = [qv for qv, mem in zip(q_values, memories) if not mem['success']]
        
        # Check Q-value distribution
        assert any(q > 0.5 for q in successful_memories), "Successful memories should have higher Q-values"
        assert any(q < 0 for q in failed_memories), "Failed memories should have lower Q-values"
    
    def teardown_method(self):
        """
        Clean up test database
        """
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

def test_ollama_availability():
    """
    Sanity check to ensure Ollama is installed and running
    """
    try:
        models = ollama.list()
        assert len(models.get('models', [])) > 0, "No Ollama models available"
    except Exception as e:
        pytest.fail(f"Ollama is not running or accessible: {e}")

if __name__ == "__main__":
    pytest.main([__file__])