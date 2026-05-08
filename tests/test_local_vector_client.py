import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the libraries before they are imported by the client
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from mapper_tools.local_vector_client import LocalVectorClient


class TestLocalVectorClient(unittest.TestCase):

    def setUp(self):
        self.config = {
            "backend": "local",
            "local_backend": "chroma",
            "store_path": "/fake/path",
            "embedding_model": {
                "model": "fake-model"
            }
        }

    @patch('mapper_tools.local_vector_client.SentenceTransformer')
    @patch('mapper_tools.local_vector_client.chromadb')
    def test_initialization_success(self, mock_chromadb, mock_sentence_transformer):
        """Test successful initialization of the LocalVectorClient with ChromaDB."""
        # Ensure the mock indicates that libraries are available
        with patch('mapper_tools.local_vector_client.CHROMA_LIBS_AVAILABLE', True):
            client = LocalVectorClient(self.config)
            self.assertIsNotNone(client)
            self.assertEqual(client.backend, "local")
            self.assertEqual(client.local_backend, "chroma")
            mock_sentence_transformer.assert_called_once_with("fake-model")
            mock_chromadb.PersistentClient.assert_called_once_with(path="/fake/path")

    def test_initialization_missing_backend(self):
        """Test initialization failure when local_backend is missing."""
        config = self.config.copy()
        del config['local_backend']
        with self.assertRaises(ValueError) as cm:
            LocalVectorClient(config)
        self.assertIn("'local_backend' must be 'chroma' or 'faiss'", str(cm.exception))

    def test_initialization_missing_store_path(self):
        """Test initialization failure when store_path is missing."""
        config = self.config.copy()
        del config['store_path']
        with self.assertRaises(ValueError) as cm:
            LocalVectorClient(config)
        self.assertIn("'store_path' must be specified", str(cm.exception))

    def test_initialization_missing_embedding_model(self):
        """Test initialization failure when embedding_model is missing."""
        config = self.config.copy()
        del config['embedding_model']
        with self.assertRaises(ValueError) as cm:
            LocalVectorClient(config)
        self.assertIn("'embedding_model' configuration with a 'model' must be provided", str(cm.exception))

    @patch('mapper_tools.local_vector_client.CHROMA_LIBS_AVAILABLE', False)
    def test_initialization_chroma_libs_unavailable(self):
        """Test initialization failure when ChromaDB libraries are not available."""
        with self.assertRaises(ImportError) as cm:
            LocalVectorClient(self.config)
        self.assertIn("libraries are required for the ChromaDB backend", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
