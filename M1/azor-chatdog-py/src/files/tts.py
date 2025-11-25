"""
Text-to-Speech generation utility using XTTS model.
"""

import threading
import time
from typing import Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from TTS.api import TTS
except ImportError:
    TTS = None


class TTSGenerator:
    """Handles text-to-speech generation using XTTS model."""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton TTS instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def load_model(cls, device: str = "cpu"):
        """
        Load the TTS model. Called once and reused.
        
        Args:
            device: Device to load model on ("cpu" or "cuda")
        """
        if cls._model is None:
            if TTS is None:
                raise RuntimeError(
                    "TTS library not installed. "
                    "Run: pip install coqui-tts"
                )
            cls._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        return cls._model
    
    def generate_audio(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "pl"
    ) -> bool:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            output_path: Path where to save the output audio file
            speaker_wav: Path to reference speaker audio (for voice cloning)
            language: Language code (default: "pl" for Polish)
            
        Returns:
            bool: True if generation succeeded, False otherwise
        """
        if not text or not text.strip():
            return False
        
        try:
            model = self.load_model()
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language
            )
            return True
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False
    
    def generate_audio_async(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "pl",
        on_complete: Optional[callable] = None
    ) -> threading.Thread:
        """
        Generate audio asynchronously in a separate thread.
        
        Args:
            text: Text to convert to speech
            output_path: Path where to save the output audio file
            speaker_wav: Path to reference speaker audio
            language: Language code
            on_complete: Callback function to call when done (passed success bool)
            
        Returns:
            threading.Thread: The thread running the generation
        """
        
        def worker():
            try:
                success = self.generate_audio(text, output_path, speaker_wav, language)
                if on_complete:
                    on_complete(success)
            except Exception as e:
                print(f"Error in async generation: {e}")
                if on_complete:
                    on_complete(False)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread
