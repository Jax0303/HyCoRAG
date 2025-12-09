import unittest
import torch
from hycorag.models.concept_distill import HybridConceptDistiller, HierarchicalConcepts

class TestConceptDistill(unittest.TestCase):
    def test_distill_shapes(self):
        hidden_dim = 64
        distiller = HybridConceptDistiller(hidden_dim=hidden_dim)
        
        # Mock inputs
        table_image = None
        table_structure = {"text": "dummy"}
        context_text = "dummy context"
        
        concepts = distiller.distill(table_image, table_structure, context_text)
        
        self.assertIsInstance(concepts, HierarchicalConcepts)
        self.assertEqual(concepts.semantic.shape[-1], hidden_dim)
        self.assertEqual(concepts.structural.shape[-1], hidden_dim)
        self.assertEqual(concepts.contextual.shape[-1], hidden_dim)
        self.assertIsNotNone(concepts.meta)

if __name__ == '__main__':
    unittest.main()
