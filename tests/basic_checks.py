import unittest
import os
import importlib.util


class TestSentimentAnalysis(unittest.TestCase):

    def test_library_torch_installed(self):
        """ Test if Pytorch library is installed """
        torch_installed = importlib.util.find_spec("torch") is not None
        self.assertTrue(torch_installed, "Pytorch is not installed")

    def test_library_datasets_installed(self):
        """ Test if datasets library is installed """
        datasets_installed = importlib.util.find_spec("datasets") is not None
        self.assertTrue(datasets_installed, "datasets library is not installed")
        
    def test_library_transformers_installed(self):
        """ Test if transformers library is installed """
        transformers_installed = importlib.util.find_spec("transformers") is not None
        self.assertTrue(transformers_installed, "transformers library is not installed")
        
    def test_library_evaluate_installed(self):
        """ Test if evaluate library is installed """
        evaluate_installed = importlib.util.find_spec("evaluate") is not None
        self.assertTrue(evaluate_installed, "evaluate library is not installed")
        
    def test_library_accelerate_installed(self):
        """ Test if accelerate library is installed """
        accelerate_installed = importlib.util.find_spec("accelerate") is not None
        self.assertTrue(accelerate_installed, "accelerate library is not installed")
        
    def test_library_mlflow_installed(self):
        """ Test if mlflow library is installed """
        mlflow_installed = importlib.util.find_spec("mlflow") is not None
        self.assertTrue(mlflow_installed, "mlflow library is not installed")
        
    def test_library_streamlit_installed(self):
        """ Test if streamlit library is installed """
        streamlit_installed = importlib.util.find_spec("streamlit") is not None
        self.assertTrue(streamlit_installed, "streamlit library is not installed")

if __name__ == '__main__':
    unittest.main()
