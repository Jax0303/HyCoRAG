import unittest
from hycorag.rag.retriever import BaselineRetriever
from hycorag.rag.pipeline import BaselineRAGPipeline, HyCoRAGPipeline, DummyLLMClient
from hycorag.models.concept_distill import HybridConceptDistiller
from hycorag.models.concept_router import ConceptRouter

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.retriever = BaselineRetriever(embedding_dim=64)
        self.retriever.index({"t1": "content1", "t2": "content2"})
        self.llm = DummyLLMClient()
        
    def test_baseline_pipeline(self):
        pipeline = BaselineRAGPipeline(self.retriever, self.llm)
        result = pipeline.run("query", top_k=1)
        
        self.assertIsNotNone(result.answer)
        self.assertEqual(len(result.retrieved_items), 1)
        self.assertEqual(result.retrieved_items[0].id, "t1") # Deterministic dummy

    def test_hycorag_pipeline(self):
        distiller = HybridConceptDistiller(hidden_dim=64)
        router = ConceptRouter(hidden_dim=64)
        pipeline = HyCoRAGPipeline(self.retriever, self.llm, distiller, router)
        
        result = pipeline.run("query", top_k=1)
        
        self.assertIsNotNone(result.answer)
        self.assertEqual(len(result.retrieved_items), 1)
        # Check if answer is generated (dummy LLM returns string)
        self.assertTrue(isinstance(result.answer, str))

if __name__ == '__main__':
    unittest.main()
