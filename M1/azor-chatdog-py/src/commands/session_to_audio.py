"""
Generate audio from the last agent response in a chat session.
"""

import threading
import time
import traceback
from typing import List, Dict
from pathlib import Path
from cli import console
from files.tts import TTSGenerator

# Path to a sample speaker WAV (for voice cloning). It's stored next to this file.
# FILE_PATH = str(Path(__file__).parent.joinpath("sample-agent.wav"))
FILE_PATH = str(Path(__file__).parent.joinpath("record_out.wav"))

def export_session_to_audio(history: List[Dict], session_id: str, assistant_name: str):
    """
    Generates an audio file from the last agent message in the session history.

    Args:
        history: List of dictionaries in the format {"role": "user|model", "parts": [{"text": "..."}]}
        session_id: The ID of the session.
        assistant_name: The name of the assistant.
    """

    if not history:
        console.print_info("Session history is empty. No audio will be generated.")
        return

    # Find the last assistant message
    last_agent_text = None
    for message in reversed(history):
        print(f"Checking message role: {message.get('role', '')}")
        print(f"Message content: {message}")
        role = message.get("role", "")
        if role == "assistant":
            if 'parts' in message and message['parts']:
                last_agent_text = message['parts'][0].get('text', '')
            break

    if not last_agent_text:
        console.print_error(f"No message from {assistant_name} found in history.")
        return

    # Prepare output path
    output_filename = f"{session_id}_audio.wav"
    
    try:
        console.print_info("\n🎙️ Initializing text-to-speech...")
        
        # Get TTS generator
        tts_gen = TTSGenerator.get_instance()
        
        # Create a flag to track completion
        generation_complete = threading.Event()
        success_flag = {'value': False}
        
        def on_complete(success: bool):
            success_flag['value'] = success
            generation_complete.set()
        
        # Verify speaker sample exists before passing to TTS (avoid system error)
        if not Path(FILE_PATH).exists():
            console.print_error(
                f"❌ Speaker sample not found: {FILE_PATH}\n"
                "Place a WAV file named 'sample-agent.wav' next to this script,\n"
                "or update `FILE_PATH` to point to a valid WAV. Aborting audio generation."
            )
            return

        # Generate audio asynchronously
        console.print_info(f"🎵 Generating audio for: '{last_agent_text[:60]}...'")
        thread = tts_gen.generate_audio_async(
            text=last_agent_text,
            output_path=output_filename,
            speaker_wav=FILE_PATH,
            language="de",
            on_complete=on_complete
        )
        
        # Wait for generation with timeout
        timeout = 300  # 5 minutes max
        if generation_complete.wait(timeout=timeout):
            if success_flag['value']:
                console.print_info(f"✅ Audio file generated successfully: {output_filename}")
            else:
                console.print_error(f"❌ Failed to generate audio file.")
        else:
            console.print_error(f"❌ Audio generation timed out after {timeout} seconds.")
            
    except ImportError:
        console.print_error(
            "❌ TTS library not installed.\n"
            "Run: pip install coqui-tts rich\n"
            "Then restart the application."
        )
    except Exception as e:
        console.print_error(f"Failed to generate audio: {e}\n{traceback.format_exc()}")
