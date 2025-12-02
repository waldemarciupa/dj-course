import json
import os
from datetime import datetime

# Test creating sample transcription files if output dir doesn't have them
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Create 3 sample transcription files for testing
test_data = [
    {
        "transcription": "This is the first test transcription with some longer text to test the truncation feature in the history list",
        "timestamp": "2025-11-29 10:15:30",
        "filename": "recording-1704067800"
    },
    {
        "transcription": "Second test recording from earlier today",
        "timestamp": "2025-11-29 09:45:15",
        "filename": "recording-1704067200"
    },
    {
        "transcription": "Another example of a transcription",
        "timestamp": "2025-11-29 08:30:00",
        "filename": "recording-1704066000"
    }
]

for data in test_data:
    filename = data["filename"]
    json_path = os.path.join(output_dir, f"{filename}.json")
    
    if not os.path.exists(json_path):
        json_data = {
            "audio_file": f"{filename}.wav",
            "transcription": data["transcription"],
            "timestamp": data["timestamp"]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Created {json_path}")

print("Test files created successfully!")
