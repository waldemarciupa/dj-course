
"""
OpenAI LLM Client Implementation
Encapsulates all OpenAI interactions.
"""

import os
import sys
from typing import Optional, List, Any, Dict
from openai import OpenAI
from dotenv import load_dotenv
from cli import console
from .openai_validation import OpenAIConfig

class ResponseMessage:
    def __init__(self, content):
        self.text = content

class OpenAIChatSessionWrapper:
    """
    Wrapper for OpenAI chat session that provides universal dictionary-based history format.
    """
    
    def __init__(self, openai_session, client, model_name):
        """
        Initialize wrapper with OpenAI chat session.
        
        Args:
            openai_session: The actual OpenAI chat session object
            client: The OpenAI client
            model_name: The model name
        """
        self.openai_session = openai_session
        self._client = client
        self.model_name = model_name
        self.history = []

    def send_message(self, text: str) -> Any:
        """
        Forwards message to OpenAI session.
        
        Args:
            text: User's message
            
        Returns:
            Response object from OpenAI
        """
        self.history.append({"role": "user", "content": text})
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=self.history
        )
        self.history.append({"role": "assistant", "content": response.choices[0].message.content})
        return ResponseMessage(response.choices[0].message.content)
    
    def get_history(self) -> List[Dict]:
        """
        Gets conversation history in universal dictionary format.
        
        Returns:
            List of dictionaries with format: {"role": "user|model", "parts": [{"text": "..."}]}
        """
        universal_history = []
        for message in self.history:
            universal_history.append({
                "role": message["role"],
                "parts": [{"text": message["content"]}]
            })
        return universal_history

class OpenAILLMClient:
    """
    Encapsulates all OpenAI interactions.
    Provides a clean interface for chat sessions, token counting, and configuration.
    """
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the OpenAI LLM client with explicit parameters.
        
        Args:
            model_name: Model to use (e.g., 'gpt-4')
            api_key: OpenAI API key
        
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")
        
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize the client during construction
        self._client = self._initialize_client()
    
    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that OpenAI client is being prepared.
        
        Returns:
            Formatted preparation message string
        """
        return "🤖 Przygotowywanie klienta OpenAI..."
    
    @classmethod
    def from_environment(cls) -> 'OpenAILLMClient':
        """
        Factory method that creates a OpenAILLMClient instance from environment variables.
        
        Returns:
            OpenAILLMClient instance initialized with environment variables
            
        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()
    
        config = OpenAIConfig(
            model_name=os.getenv('MODEL_NAME', 'gpt-4'),
            openai_api_key=os.getenv('OPENAI_API_KEY', '')
        )
        
        return cls(model_name=config.model_name, api_key=config.openai_api_key)
    
    def _initialize_client(self) -> OpenAI:
        """
        Initializes the OpenAI client.
        
        Returns:
            Initialized OpenAI client
            
        Raises:
            SystemExit: If client initialization fails
        """
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta OpenAI: {e}")
            sys.exit(1)
    
    def create_chat_session(self, 
                          system_instruction: str, 
                          history: Optional[List[Dict]] = None,
                          thinking_budget: int = 0) -> OpenAIChatSessionWrapper:
        """
        Creates a new chat session with the specified configuration.
        
        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Thinking budget for the model (not used by OpenAI)
            
        Returns:
            OpenAIChatSessionWrapper with universal dictionary-based interface
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        
        openai_history = []
        if history:
            for entry in history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    text = entry['parts'][0].get('text', '') if entry['parts'] else ''
                    if text:
                        openai_history.append({"role": entry["role"], "content": text})

        if system_instruction:
            openai_history.insert(0, {"role": "system", "content": system_instruction})

        
        session = OpenAIChatSessionWrapper(openai_history, self._client, self.model_name)
        session.history = openai_history
        return session
    
    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Counts tokens for the given conversation history.
        
        Args:
            history: Conversation history in universal dict format
            
        Returns:
            Total token count
        """
        # This is a simplified token counting method. For more accuracy, consider using the `tiktoken` library.
        if not history:
            return 0
        
        token_count = 0
        for entry in history:
            if isinstance(entry, dict) and 'parts' in entry:
                text = entry['parts'][0].get('text', '') if entry['parts'] else ''
                token_count += len(text.split())
        return token_count

    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name
    
    def is_available(self) -> bool:
        """
        Checks if the LLM service is available and properly configured.
        
        Returns:
            True if client is properly initialized and has API key
        """
        return self._client is not None and bool(self.api_key)
    
    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and masked API key.
        
        Returns:
            Formatted message string for display
        """
        if len(self.api_key) <= 8:
            masked_key = "****"
        else:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        
        return f"✅ Klient OpenAI gotowy do użycia (Model: {self.model_name}, Key: {masked_key})"
    
    @property
    def client(self):
        """
        Provides access to the underlying OpenAI client.
        """
        return self._client
