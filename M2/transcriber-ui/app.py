import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import pyaudio
import wave
import os
import time
import threading
import queue
import sys
import logging
import logging.handlers
import json
from typing import TextIO
from datetime import datetime
import pygame

# --- Global Configuration ---
APP_TITLE = "Azor Transcriber"
# Set to True to print output to the console (standard output/stderr).
VERBOSE = False
LOG_FILENAME = "transcriber.log"

# --- Logging Setup ---
class StreamToLogger(TextIO):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    This captures stdout/stderr, including print() statements.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Handle buffer and write line by line
        for line in buf.rstrip().splitlines():
            # Check if the line is not empty (prevents logging empty lines from print())
            if line.strip():
                self.logger.log(self.level, line.strip())

    def flush(self):
        # Required by TextIO interface, but we flush line-by-line in write
        pass

# Configure the global logger BEFORE application startup
def setup_logging():
    """Con gures the logging system to save all output to a le and optionally to console."""
    os.makedirs('output', exist_ok=True)
    
    # 1. Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Capture everything from INFO level up

    # 2. File Handler (Always active)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME, 
        maxBytes=1024*1024*5, # 5 MB per file
        backupCount=5,
        encoding='utf-8'
    )
    # Define a simple formatter for the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. Console Handler (Only active if VERBOSE is True)
    if VERBOSE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 4. Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(root_logger, logging.INFO)
    sys.stderr = StreamToLogger(root_logger, logging.ERROR)

setup_logging()
logging.info("Application initialization started.")

# --- Whisper Dependencies ---
# Ensure you have installed: pip install torch transformers librosa
# (Librosa might require ffmpeg)
try:
    import torch
    from transformers import pipeline
except ImportError:
    logging.error("ERROR: 'transformers' or 'torch' libraries not found.")
    logging.error("Install them using: pip install torch transformers")
    exit()

# === 1. Transcription Configuration ===
MODEL_NAME = "openai/whisper-tiny"

def output_filename()  -> str:
    """Generates output filename for transcription results."""
    os.makedirs('output', exist_ok=True)
    return f"output/recording-{int(time.time())}.wav"

def transcribe_audio(audio_path: str, model_name: str) -> str:
    """
    Loads the Whisper model and transcribes the audio file.
    This function is blocking and should be run in a separate thread.
    """
    try:
        logging.info(f"Loading model: {model_name}...")
        # Initialize pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            device=device
        )

        logging.info(f"Starting transcription for file: {audio_path}...")
        result = asr_pipeline(audio_path)
        
        transcription = result["text"].strip()
        
        logging.info("Transcription finished.")
        return transcription

    except FileNotFoundError:
        logging.error(f"ERROR: Audio file not found at path: {audio_path}")
        return f"ERROR: Audio file not found at path: {audio_path}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
        return f"An unexpected error occurred during transcription: {e}"


# === 2. Recording Configuration ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Standard for speech models (Whisper)
MAX_RECORD_DURATION = 30 # Maximum recording length in seconds

# === 3. Tkinter GUI Application ===
class AudioRecorderApp:
    def __init__(self, master):
        self.master = master
        master.title(APP_TITLE)
        try:
            self.master.tk.call('wm', 'iconname', self.master._w, APP_TITLE)
        except tk.TclError:
            self.master.wm_iconname(APP_TITLE)
        master.geometry("600x450")
        master.config(bg="#121212")
        style = ttk.Style()
        style.theme_use('default')
        style.configure('SmallDelete.TButton',
            background='#444444',
            foreground='white',
            font=('Arial', 9),
            padding=(6, 2, 6, 2),
            borderwidth=0,
            relief='flat'
        )
        style.map('SmallDelete.TButton',
            background=[('active', '#666666')],
            foreground=[('active', 'white')]
        )
        style.configure('TNotebook', background='#121212', borderwidth=0)
        style.configure('TNotebook.Tab', background='#1E1E1E', foreground='white', borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#0F0F0F')], foreground=[('selected', 'white')])
        style.configure('Dark.TButton',
            background='#333333',
            foreground='white',
            font=('Arial', 14),
            bordercolor='#333333',
            borderwidth=0,
            focuscolor='#333333',
            padding=(20, 10, 20, 10)
        )
        style.map('Dark.TButton',
            background=[('active', '#555555'), ('disabled', '#333333')],
            foreground=[('active', 'white')]
        )
        logging.info("GUI initialization started.")
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            logging.critical(f"Could not initialize PyAudio: {e}. Destroying GUI.")
            messagebox.showerror("PyAudio Error", f"Could not initialize PyAudio: {e}\nDo you have 'portaudio' installed?")
            master.destroy()
            return
        self.frames = []
        self.stream = None
        self.recording = False
        self.start_time = None
        self.record_timer_id = None
        self.transcription_queue = queue.Queue()
        self.notebook = ttk.Notebook(master, style='TNotebook')
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)
        self.transcriber_frame = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.transcriber_frame, text='Transcriber')
        self.history_frame = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.history_frame, text='Transcription History')
        self.history_container = tk.Frame(self.history_frame, bg="#121212")
        self.history_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Row for sidebar and content labels
        self.history_labels_row = tk.Frame(self.history_container, bg="#121212")
        self.history_labels_row.pack(fill=tk.X, padx=10, pady=(10,0))
        tk.Label(self.history_labels_row, text="Saved Transcriptions", font=('Arial', 12, 'bold'), fg='white', bg="#0F0F0F", anchor="w").pack(side=tk.LEFT, padx=(0,20))
        tk.Label(self.history_labels_row, text="Transcription Details:", font=('Arial', 12, 'bold'), fg='white', bg="#121212", anchor="w").pack(side=tk.LEFT)
        self.history_sidebar = tk.Frame(self.history_container, bg="#0F0F0F", width=300)
        self.history_sidebar.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        self.history_sidebar.pack_propagate(False)
        self.history_items_frame = tk.Frame(self.history_sidebar, bg="#0F0F0F")
        self.history_items_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        self.history_content = tk.Frame(self.history_container, bg="#121212")
        self.history_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # (Transcription Details label moved to labels row above)
        self.history_display = tk.Text(self.history_content,
            height=15,
            wrap=tk.WORD,
            font=('Arial', 11),
            relief=tk.SUNKEN,
            bg='#1E1E1E',
            fg='white',
            insertbackground='white',
            state=tk.DISABLED
        )
        self.history_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.history_button_frame = tk.Frame(self.history_content, bg="#121212")
        self.history_button_frame.pack(pady=10, padx=0, anchor=tk.CENTER)
        self.history_play_button = ttk.Button(self.history_button_frame,
            text="▶ Play Audio",
            command=self.play_selected_audio,
            style='Dark.TButton',
            state=tk.DISABLED
        )
        self.history_play_button.pack()
        self.selected_transcription_data = None
        self.refresh_transcription_history()
        self.settings_frame = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.settings_frame, text='Settings')
        tk.Label(self.settings_frame, text="Under construction...", font=('Arial', 18), fg='gray', bg="#121212").pack(pady=50)
        self.record_button = ttk.Button(self.transcriber_frame,
            text="Record",
            command=self.toggle_recording,
            style='Dark.TButton'
        )
        self.record_button.pack(pady=20, fill=tk.X, padx=20)
        self.transcription_display = tk.Text(self.transcriber_frame,
            height=10,
            wrap=tk.WORD,
            font=('Arial', 11),
            relief=tk.SUNKEN,
            bg='#1E1E1E',
            fg='white',
            insertbackground='white',
            state=tk.DISABLED
        )
        self.transcription_display.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.transcription_display.config(state=tk.NORMAL)
        self.transcription_display.insert(tk.END, "Transcribed text will appear here. Select it to copy.")
        self.transcription_display.config(state=tk.DISABLED)
        self.exit_button = ttk.Button(master,
            text="Exit",
            command=self.on_closing,
            style='Dark.TButton'
        )
        self.exit_button.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.after(100, self.check_transcription_queue)
        logging.info("GUI initialized successfully.")
    def __init__(self, master):
        self.master = master
        
        # 1. Set application title (window title)
        master.title(APP_TITLE)
        
        # 2. Set the application name for the OS/taskbar
        # This is cross-platform attempt to set the application name
        try:
            # For macOS and some X11 environments
            self.master.tk.call('wm', 'iconname', self.master._w, APP_TITLE)
        except tk.TclError:
            # Standard method, usually works on Windows/Linux
            self.master.wm_iconname(APP_TITLE)
            
        master.geometry("600x450") # Slightly larger window
        master.config(bg="#121212") # Set dark background for root

        # --- TKINTER WIDGET STYLES (ttk) ---
        style = ttk.Style()
        style.theme_use('default') 

        # Custom style for small delete button (moved here)
        style.configure('SmallDelete.TButton',
            background='#888888',
            foreground='white',
            font=('Arial', 9),
            padding=(6, 2, 6, 2),
            borderwidth=0,
            relief='flat'
        )
        style.map('SmallDelete.TButton',
            background=[('active', '#AAAAAA')],
            foreground=[('active', 'white')]
        )

        # Configure the dark background for the Notebook tabs
        style.configure('TNotebook', background='#121212', borderwidth=0)
        style.configure('TNotebook.Tab', background='#1E1E1E', foreground='white', borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#0F0F0F')], foreground=[('selected', 'white')])

        # 1. Define new style for dark gray buttons
        style.configure('Dark.TButton',
                        background='#333333',    
                        foreground='white',     
                        font=('Arial', 14),
                        bordercolor='#333333',
                        borderwidth=0,
                        focuscolor='#333333',
                        padding=(20, 10, 20, 10) 
                       )
        
        # 2. Define button appearance in different states (active/disabled)
        style.map('Dark.TButton',
                  background=[('active', '#555555'), # Lighter gray for hover/active state
                              ('disabled', '#333333')], # Disabled state uses the default background
                 )

        logging.info("GUI initialization started.")

        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            logging.critical(f"Could not initialize PyAudio: {e}. Destroying GUI.")
            messagebox.showerror("PyAudio Error", f"Could not initialize PyAudio: {e}\nDo you have 'portaudio' installed?")
            master.destroy()
            return
            
        self.frames = []
        self.stream = None
        self.recording = False
        self.start_time = None
        self.record_timer_id = None 

        # Queue for inter-thread communication
        self.transcription_queue = queue.Queue()
        
        # --- TAB MENU SETUP (Notebook) ---
        self.notebook = ttk.Notebook(master, style='TNotebook')
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)

        # 1. Transcriber Tab
        self.transcriber_frame = tk.Frame(self.notebook, bg="#121212") # Set dark background for frame
        self.notebook.add(self.transcriber_frame, text='Transcriber')

        # 2. History Tab
        self.history_frame = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.history_frame, text='Transcription History')
        
        # Container frame for sidebar + content layout
        self.history_container = tk.Frame(self.history_frame, bg="#121212")
        self.history_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Sidebar (Left) ---
        self.history_sidebar = tk.Frame(self.history_container, bg="#0F0F0F", width=300)
        self.history_sidebar.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        self.history_sidebar.pack_propagate(False)
        
        tk.Label(self.history_sidebar, text="Saved Transcriptions", font=('Arial', 12, 'bold'), fg='white', bg="#0F0F0F").pack(pady=(10, 10), padx=10)
        
        # Custom sidebar for transcription items (no scrollbar, per-item delete)
        self.history_items_frame = tk.Frame(self.history_sidebar, bg="#0F0F0F")
        self.history_items_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # --- Content Area (Right) ---
        self.history_content = tk.Frame(self.history_container, bg="#121212")
        self.history_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(self.history_content, text="Transcription Details:", font=('Arial', 12, 'bold'), fg='white', bg="#121212").pack(pady=(0, 10), anchor=tk.W)
        
        # Text display for transcription content
        self.history_display = tk.Text(self.history_content, 
                                       height=15, 
                                       wrap=tk.WORD, 
                                       font=('Arial', 11),
                                       relief=tk.SUNKEN, 
                                       bg='#1E1E1E', 
                                       fg='white', 
                                       insertbackground='white', 
                                       state=tk.DISABLED 
                                      )
        self.history_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Play button frame
        self.history_button_frame = tk.Frame(self.history_content, bg="#121212")
        self.history_button_frame.pack(pady=10, padx=0, anchor=tk.CENTER)
        
        self.history_play_button = ttk.Button(self.history_button_frame, 
                                              text="▶ Play Audio", 
                                              command=self.play_selected_audio,
                                              style='Dark.TButton',
                                              state=tk.DISABLED)
        self.history_play_button.pack()

            # Delete button for transcription
        # (Delete button removed from content side)
        
        # Store metadata for selected transcription
        self.selected_transcription_data = None
        
        # Load transcriptions on startup
        self.refresh_transcription_history()


        # 3. Settings Tab
        self.settings_frame = tk.Frame(self.notebook, bg="#121212") 
        self.notebook.add(self.settings_frame, text='Settings')

        # Content for Settings Tab
        tk.Label(self.settings_frame, text="Under construction...", font=('Arial', 18), fg='gray', bg="#121212").pack(pady=50)


        # --- Transcriber Tab Elements ---
        
        # Record Button
        self.record_button = ttk.Button(self.transcriber_frame, 
                                        text="Record", 
                                        command=self.toggle_recording, 
                                        style='Dark.TButton')
        self.record_button.pack(pady=20, fill=tk.X, padx=20) 

        # Transcribed Text Display (Read-only Text widget)
        self.transcription_display = tk.Text(self.transcriber_frame, 
                                             height=10, 
                                             wrap=tk.WORD, 
                                             font=('Arial', 11),
                                             relief=tk.SUNKEN, 
                                             bg='#1E1E1E', 
                                             fg='white', 
                                             insertbackground='white', 
                                             state=tk.DISABLED 
                                             )
        self.transcription_display.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Initial text insertion for tk.Text
        self.transcription_display.config(state=tk.NORMAL)
        self.transcription_display.insert(tk.END, "Transcribed text will appear here. Select it to copy.")
        self.transcription_display.config(state=tk.DISABLED)


        # Exit Button
        self.exit_button = ttk.Button(master, 
                                      text="Exit", 
                                      command=self.on_closing,
                                      style='Dark.TButton')
        self.exit_button.pack(pady=10)

        # Handle window closing
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the loop checking the queue
        self.master.after(100, self.check_transcription_queue)
        logging.info("GUI initialized successfully.")
    
    def copy_to_clipboard(self, text: str):
        """Copies the given text to the system clipboard."""
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        logging.info("Transcription copied to clipboard.")

    def load_transcription_files(self):
        """Load all transcription files from output directory."""
        transcriptions = []
        output_dir = 'output'
        
        if not os.path.exists(output_dir):
            return transcriptions
        
        # Get all JSON files
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        json_files.sort(reverse=True)  # Most recent first
        
        for json_file in json_files:
            try:
                json_path = os.path.join(output_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Get corresponding WAV file
                wav_file = json_file.replace('.json', '.wav')
                wav_path = os.path.join(output_dir, wav_file)
                
                if os.path.exists(wav_path):
                    transcriptions.append({
                        'json_file': json_file,
                        'json_path': json_path,
                        'wav_file': wav_file,
                        'wav_path': wav_path,
                        'transcription': data.get('transcription', ''),
                        'timestamp': data.get('timestamp', ''),
                        'audio_file': data.get('audio_file', '')
                    })
            except Exception as e:
                logging.error(f"Error loading transcription file {json_file}: {e}")
        
        return transcriptions

    def format_timestamp(self, timestamp_str):
        """Convert timestamp string to readable format (e.g., '3 April 3:03 pm')."""
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            # Format: "29 November 3:03 pm" - cross-platform
            day = str(dt.day)
            month = dt.strftime('%B')
            hour = dt.hour % 12 if dt.hour % 12 != 0 else 12
            minute = dt.strftime('%M')
            am_pm = 'am' if dt.hour < 12 else 'pm'
            return f"{day} {month} {hour}:{minute} {am_pm}"
        except:
            return timestamp_str

    def truncate_text(self, text, max_length=40):
        """Truncate text and add ellipsis if needed."""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def refresh_transcription_history(self):
        """Load and display transcription history in the sidebar as custom items."""
        for widget in self.history_items_frame.winfo_children():
            widget.destroy()
        transcriptions = self.load_transcription_files()
        self.transcriptions_list = transcriptions
        for idx, trans in enumerate(transcriptions):
            item_frame = tk.Frame(self.history_items_frame, bg="#1E1E1E", pady=2)
            item_frame.pack(fill=tk.X, expand=False, pady=2)
            truncated = self.truncate_text(trans['transcription'])
            label = tk.Label(item_frame, text=truncated, bg="#1E1E1E", fg="white", font=("Arial", 10), anchor="w")
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            label.bind("<Button-1>", lambda e, i=idx: self.on_transcription_item_click(i))
            del_btn = ttk.Button(
                item_frame,
                text="x",  # Only trash icon
                style='SmallDelete.TButton',
                width=1,  # Fixed width for icon only, prevents shrinking
                command=lambda i=idx: self.delete_transcription_by_index(i)
            )
            del_btn.pack(side=tk.RIGHT, padx=(4,2), pady=2, anchor='e')
        logging.info(f"Loaded {len(transcriptions)} transcription(s).")

    def on_transcription_item_click(self, index):
        """Handle click on a transcription item label."""
        if index < len(self.transcriptions_list):
            self.selected_transcription_data = self.transcriptions_list[index]
            trans_data = self.selected_transcription_data
            display_content = f"File: {trans_data['audio_file']}\n"
            display_content += f"Date: {self.format_timestamp(trans_data['timestamp'])}\n"
            display_content += f"\n{'='*50}\n\n"
            display_content += trans_data['transcription']
            self.history_display.config(state=tk.NORMAL)
            self.history_display.delete('1.0', tk.END)
            self.history_display.insert(tk.END, display_content)
            self.history_display.config(state=tk.DISABLED)
            self.history_play_button.config(state=tk.NORMAL)
        else:
            self.selected_transcription_data = None
            self.history_play_button.config(state=tk.DISABLED)

    def delete_transcription_by_index(self, index):
        """Delete transcription and files by index."""
        if index >= len(self.transcriptions_list):
            return
        trans_data = self.transcriptions_list[index]
        json_path = trans_data.get('json_path')
        wav_path = trans_data.get('wav_path')
        confirm = messagebox.askyesno("Delete Transcription", "Are you sure you want to delete this transcription? This cannot be undone.")
        if not confirm:
            return
        errors = []
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception as e:
                errors.append(f"Could not delete JSON file: {e}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                errors.append(f"Could not delete WAV file: {e}")
        self.selected_transcription_data = None
        self.history_play_button.config(state=tk.DISABLED)
        self.history_display.config(state=tk.NORMAL)
        self.history_display.delete('1.0', tk.END)
        self.history_display.config(state=tk.DISABLED)
        self.refresh_transcription_history()
        if errors:
            messagebox.showerror("Delete Error", "\n".join(errors))
        else:
            messagebox.showinfo("Deleted", "Transcription deleted successfully.")

    def on_transcription_select(self, event):
        """Handle transcription selection from listbox."""
        selection = self.history_listbox.curselection()
        if not selection:
            self.selected_transcription_data = None
            self.history_play_button.config(state=tk.DISABLED)
            self.history_delete_button.config(state=tk.DISABLED)
            return

        index = selection[0]
        if index < len(self.transcriptions_list):
            self.selected_transcription_data = self.transcriptions_list[index]
            trans_data = self.selected_transcription_data
            display_content = f"File: {trans_data['audio_file']}\n"
            display_content += f"Date: {self.format_timestamp(trans_data['timestamp'])}\n"
            display_content += f"\n{'='*50}\n\n"
            display_content += trans_data['transcription']
            self.history_display.config(state=tk.NORMAL)
            self.history_display.delete('1.0', tk.END)
            self.history_display.insert(tk.END, display_content)
            self.history_display.config(state=tk.DISABLED)
            self.history_play_button.config(state=tk.NORMAL)
            self.history_delete_button.config(state=tk.NORMAL)
            logging.info(f"Selected transcription: {trans_data['audio_file']}")
        else:
            self.selected_transcription_data = None
            self.history_play_button.config(state=tk.DISABLED)
            self.history_delete_button.config(state=tk.DISABLED)
    def delete_selected_transcription(self):
        """Delete the selected transcription and its files."""
        if not self.selected_transcription_data:
            messagebox.showwarning("No Selection", "Please select a transcription to delete.")
            return

        trans_data = self.selected_transcription_data
        json_path = trans_data.get('json_path')
        wav_path = trans_data.get('wav_path')

        confirm = messagebox.askyesno("Delete Transcription", "Are you sure you want to delete this transcription? This cannot be undone.")
        if not confirm:
            return

        errors = []
        # Delete JSON file
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception as e:
                errors.append(f"Could not delete JSON file: {e}")

        # Delete WAV file
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                errors.append(f"Could not delete WAV file: {e}")

        # Clear selection and refresh history
        self.selected_transcription_data = None
        self.history_play_button.config(state=tk.DISABLED)
        self.history_delete_button.config(state=tk.DISABLED)
        self.history_display.config(state=tk.NORMAL)
        self.history_display.delete('1.0', tk.END)
        self.history_display.config(state=tk.DISABLED)
        self.refresh_transcription_history()

        if errors:
            messagebox.showerror("Delete Error", "\n".join(errors))
        else:
            messagebox.showinfo("Deleted", "Transcription deleted successfully.")

    def play_selected_audio(self):
        """Play the selected audio file."""
        if not self.selected_transcription_data:
            messagebox.showwarning("No Selection", "Please select a transcription to play.")
            return
        
        wav_path = self.selected_transcription_data['wav_path']
        
        if not os.path.exists(wav_path):
            messagebox.showerror("File Not Found", f"Audio file not found: {wav_path}")
            return
        
        try:
            # Initialize pygame mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Stop any currently playing sound
            pygame.mixer.stop()
            
            # Load and play the audio
            sound = pygame.mixer.Sound(wav_path)
            sound.play()
            
            logging.info(f"Playing audio: {wav_path}")
            
            # Update button text to show playback status
            self.history_play_button.config(text="⏸ Playing...")
            self.master.after(int(sound.get_length() * 1000), self.playback_finished)
            
        except Exception as e:
            logging.error(f"Error playing audio file: {e}")
            messagebox.showerror("Playback Error", f"Could not play audio file: {e}")

    def playback_finished(self):
        """Called when audio playback finishes."""
        self.history_play_button.config(text="▶ Play Audio")
        logging.info("Audio playback finished.")

    def toggle_recording(self):
        """Toggles the recording state (start/stop)."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Starts the audio recording process."""
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        logging.info("Recording started.")
        
        try:
            self.stream = self.p.open(format=FORMAT,
                                     channels=CHANNELS,
                                     rate=RATE,
                                     input=True,
                                     frames_per_buffer=CHUNK)

            # Update button text to show status
            self.record_button.config(text="Stop Recording") 
            
            # Update text display
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, "Recording in progress... (max 30s)")
            self.transcription_display.config(state=tk.DISABLED)
            
            self.read_chunk()
            # Set a timer for automatic stop
            self.record_timer_id = self.master.after(MAX_RECORD_DURATION * 1000, self.auto_stop_recording)

        except Exception as e:
            self.recording = False
            self.record_button.config(text="Record", state=tk.NORMAL) 
            logging.error(f"Microphone stream error on start: {e}")
            messagebox.showerror("Audio Error", f"Could not open microphone stream: {e}\nCheck your microphone connection and permissions.")
            if self.record_timer_id:
                self.master.after_cancel(self.record_timer_id)
                self.record_timer_id = None
            
    def read_chunk(self):
        """Reads one audio chunk and schedules the next call."""
        if self.recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                self.master.after(1, self.read_chunk) 
            except IOError as e:
                logging.error(f"Stream read IOError: {e}")
                self.stop_recording()

    def auto_stop_recording(self):
        """Automatically stops recording after MAX_RECORD_DURATION expires."""
        if self.recording:
            logging.info(f"Automatic stop triggered after {MAX_RECORD_DURATION} seconds.")
            self.stop_recording()
            messagebox.showinfo("Recording Finished", f"The recording was stopped automatically after {MAX_RECORD_DURATION} seconds. Starting transcription...")

    def stop_recording(self):
        """Stops the stream, saves the file, and starts the transcription thread."""
        if not self.recording:
            return

        self.recording = False
        
        if self.record_timer_id:
            self.master.after_cancel(self.record_timer_id)
            self.record_timer_id = None

        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logging.info("Audio stream closed.")

        WAVE_OUTPUT_FILENAME = output_filename()
        
        # Update button status for user feedback
        self.record_button.config(text="Saving...", state=tk.DISABLED) 
        self.master.update_idletasks()

        # Save to WAVE file
        try:
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            logging.info(f"File saved successfully to {WAVE_OUTPUT_FILENAME}")
            
            self.record_button.config(text="Transcribing...")
            
            # Update text in read-only Text widget
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, "Transcription in progress (this may take a while)...")
            self.transcription_display.config(state=tk.DISABLED)
            
            # === START TRANSCRIPTION IN A THREAD ===
            transcription_thread = threading.Thread(
                target=self.run_transcription,
                args=(WAVE_OUTPUT_FILENAME,),
                daemon=True
            )
            transcription_thread.start()
            logging.info("Transcription thread started.")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save WAVE file: {e}")
            self.record_button.config(text="Record", state=tk.NORMAL) 
            logging.error(f"Error saving wave file: {e}", exc_info=True)

    def run_transcription(self, audio_path):
        """
        Method executed in a separate thread. 
        Calls transcription and puts the result in the queue.
        """
        logging.info(f"Running transcription for {audio_path} in thread: {threading.get_ident()}")
        transcription = transcribe_audio(audio_path, MODEL_NAME)
        
        # Save transcription to JSON file
        if "ERROR" not in transcription:
            try:
                json_filename = audio_path.replace('.wav', '.json')
                json_data = {
                    "audio_file": os.path.basename(audio_path),
                    "transcription": transcription,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(json_data, json_file, ensure_ascii=False, indent=2)
                logging.info(f"Transcription saved to {json_filename}")
            except Exception as e:
                logging.error(f"Error saving JSON file: {e}", exc_info=True)
        
        self.transcription_queue.put(transcription)

    def check_transcription_queue(self):
        """
        Checks the queue for transcription results.
        Run in the main GUI thread.
        """
        try:
            result = self.transcription_queue.get(block=False)
            
            # 1. Update Transcriber tab (main output)
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, result)
            self.transcription_display.config(state=tk.DISABLED)
            
            # 2. Refresh history tab
            self.refresh_transcription_history()
            
            if "ERROR" in result:
                logging.warning("Transcription failed with error message.")
                messagebox.showerror("Transcription Failed", "Transcription returned an error. Check logs for details.")
            else:
                # Copy to clipboard upon successful transcription
                self.copy_to_clipboard(result) 
                
            self.record_button.config(text="Record", state=tk.NORMAL) # Return to normal state

        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_transcription_queue)

    def on_closing(self):
        """Handles clean application shutdown."""
        logging.info("Closing application...")
        if self.recording:
            self.stop_recording() 
        
        # Terminate PyAudio
        if self.p:
            self.p.terminate()
        
        self.master.destroy()
        logging.info("Application destroyed.")

# --- Application Startup ---
if __name__ == "__main__":
    logging.info("Whisper model loading might take a moment on first launch...")
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()
