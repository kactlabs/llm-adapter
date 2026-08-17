"""
@author: Raja CSP Raman

source:
    ?
"""

import os
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Adapter Pattern - Abstract base class for LLM adapters
class LLMAdapter(ABC):
    """Abstract adapter for different LLM providers"""
    
    @abstractmethod
    def get_client(self):
        """Return configured LLM client"""
        pass


# Concrete Adapters for different LLM providers
class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama LLM provider"""
    
    def get_client(self):
        from langchain_ollama import ChatOllama
        
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"[DEBUG][OllamaAdapter] Model: {model}")
        print(f"[DEBUG][OllamaAdapter] Base URL: {base_url}")
        print(f"[DEBUG][OllamaAdapter] Temperature: 0.7, num_predict: 512, timeout: 120")
        
        # Check if the model is a thinking/reasoning model that might need special config
        thinking_models = ['qwen3', 'deepseek-r1', 'qwq']
        is_thinking_model = any(tm in model.lower() for tm in thinking_models)
        if is_thinking_model:
            print(f"[DEBUG][OllamaAdapter] Detected potential thinking model: {model}")
            print(f"[DEBUG][OllamaAdapter] Note: Thinking models may return content in 'thinking' field instead of main content")
        
        # Thinking models need more tokens since they use tokens for reasoning before answering
        if is_thinking_model:
            num_predict = 4096
            print(f"[DEBUG][OllamaAdapter] Using higher num_predict={num_predict} for thinking model")
        else:
            num_predict = 512
        
        client = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.7,
            num_predict=num_predict,
            timeout=180  # Higher timeout for thinking models
        )
        
        print(f"[DEBUG][OllamaAdapter] ChatOllama client created successfully")
        print(f"[DEBUG][OllamaAdapter] Client type: {type(client).__name__}")
        
        return client


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI LLM provider"""
    
    def get_client(self):
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=512,
            timeout=120  # Increased timeout to 120 seconds
        )


class LlamaCppAdapter(LLMAdapter):
    """Adapter for llama.cpp LLM provider (direct HTTP, no langchain-openai dependency)"""
    
    def __init__(self, base_url="http://127.0.0.1:8080/v1"):
        self.base_url = base_url
    
    def _check_server_health(self):
        """Check if llama.cpp server is running"""
        try:
            # Try to reach the health endpoint or models endpoint
            health_url = self.base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            try:
                # Fallback: try the models endpoint
                models_url = f"{self.base_url}/models"
                response = requests.get(models_url, timeout=5)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False
    
    def get_client(self):
        """Return a simple HTTP client wrapper for llama.cpp"""
        if not self._check_server_health():
            raise ConnectionError(f"llama.cpp server is not running at {self.base_url}. Please start your llama.cpp server first.")
        
        # Return a simple wrapper that mimics langchain interface
        return LlamaCppClient(self.base_url)


class LlamaCppClient:
    """Simple HTTP client for llama.cpp that mimics langchain interface"""
    
    def __init__(self, base_url):
        self.base_url = base_url
    
    def invoke(self, prompt):
        """Send request to llama.cpp server"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract content from response
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Return object with content attribute (like langchain)
            class Response:
                def __init__(self, content):
                    self.content = content
            
            return Response(content)
            
        except Exception as e:
            raise Exception(f"llama.cpp request failed: {e}")


class GeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini LLM provider"""
    
    def get_client(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=512,
            timeout=120
        )


# Factory Pattern - Creates appropriate LLM adapter based on provider
class LLMFactory:
    """Factory for creating LLM adapters based on provider type"""
    
    _adapters = {
        "ollama": OllamaAdapter,
        "openai": OpenAIAdapter,
        "llama.cpp": LlamaCppAdapter,
        "gemini": GeminiAdapter,
    }
    
    @classmethod
    def create_adapter(cls, provider: str) -> LLMAdapter:
        """Create and return appropriate LLM adapter"""
        provider = provider.lower().strip()
        
        if provider not in cls._adapters:
            available_providers = ", ".join(cls._adapters.keys())
            raise ValueError(f"Unsupported LLM provider: {provider}. Available providers: {available_providers}")
        
        return cls._adapters[provider]()
    
    @classmethod
    def register_adapter(cls, provider: str, adapter_class: type):
        """Register a new LLM adapter (for extensibility)"""
        cls._adapters[provider] = adapter_class


def get_llm():
    """Get LLM client based on environment configuration"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"\n{'='*60}")
    print(f"[DEBUG][get_llm] Provider: {provider}")
    print(f"[DEBUG][get_llm] OLLAMA_MODEL env: {os.getenv('OLLAMA_MODEL', 'NOT SET')}")
    print(f"[DEBUG][get_llm] OLLAMA_BASE_URL env: {os.getenv('OLLAMA_BASE_URL', 'NOT SET')}")
    print(f"{'='*60}\n")
    
    adapter = LLMFactory.create_adapter(provider)
    print(f"[DEBUG][get_llm] Adapter created: {type(adapter).__name__}")
    client = adapter.get_client()
    print(f"[DEBUG][get_llm] Client ready: {type(client).__name__}")
    return client


def get_llm_info():
    """Get current LLM provider and model info"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    elif provider == "llama.cpp":
        model = _get_llamacpp_model()
    else:
        model = "unknown"
    return {"provider": provider, "model": model}


def _get_llamacpp_model():
    """Get model name from llama.cpp server's /models endpoint"""
    base_url = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080/v1")
    try:
        response = requests.get(f"{base_url}/models", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id", "llama.cpp local")
    except Exception:
        pass
    return os.getenv("LLAMA_CPP_MODEL", "llama.cpp local")


# Tool Calling Functions




