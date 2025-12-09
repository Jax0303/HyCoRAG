import unittest
from hycorag.data.datasets import TableQADataset

class TestDatasets(unittest.TestCase):
    def test_dataset_initialization(self):
        data = [{"question": "q1", "answer": "a1"}]
        dataset = TableQADataset(data)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["question"], "q1")

if __name__ == "__main__":
    unittest.main()
