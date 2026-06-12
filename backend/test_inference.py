import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.generator import invoke_llm, HAS_GEMINI, HAS_OPENAI

class TestLLMFallback(unittest.TestCase):

    @patch('utils.generator.ChatGoogleGenerativeAI')
    def test_gemini_success(self, mock_gemini):
        # Setup mock Gemini response
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="Gemini success response")
        mock_gemini.return_value = mock_instance

        # Mock env keys
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            # Test calling invoke_llm
            res = invoke_llm("test prompt")
            self.assertEqual(res.content, "Gemini success response")
            # Verify it tried the preferred model
            mock_gemini.assert_any_call(model="gemini-2.5-flash", google_api_key="test-key", temperature=0.0)

    @patch('utils.generator.ChatGoogleGenerativeAI')
    @patch('utils.generator.ChatOpenAI')
    def test_gemini_fails_openai_success(self, mock_openai, mock_gemini):
        # Gemini fails on all models
        mock_gemini.side_effect = Exception("Gemini API Error")
        
        # OpenAI succeeds
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="OpenAI success response")
        mock_openai.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OPENAI_API_KEY": "test-openai-key"}):
            res = invoke_llm("test prompt")
            self.assertEqual(res.content, "OpenAI success response")
            # Verify OpenAI initialization
            mock_openai.assert_called_with(model="gpt-4o-mini", api_key="test-openai-key", temperature=0.0)

    @patch('utils.generator.ChatGoogleGenerativeAI')
    @patch('utils.generator.ChatOpenAI')
    @patch('utils.generator.ChatOllama')
    def test_all_fail_throws_friendly_error(self, mock_ollama, mock_openai, mock_gemini):
        # All models throw exception
        mock_gemini.side_effect = Exception("Gemini offline")
        mock_openai.side_effect = Exception("OpenAI quota exceeded")
        mock_ollama.side_effect = Exception("Ollama connection refused")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OPENAI_API_KEY": "test-openai-key"}):
            with self.assertRaises(RuntimeError) as context:
                invoke_llm("test prompt")
            
            error_msg = str(context.exception)
            self.assertIn("AI Inference Failure", error_msg)
            self.assertIn("Gemini API Key", error_msg)
            self.assertIn("OpenAI API Key", error_msg)
            self.assertIn("Local Ollama", error_msg)
            self.assertIn("Ollama connection refused", error_msg)

if __name__ == '__main__':
    print("Running LLM Fallback Verification Tests...")
    unittest.main()
