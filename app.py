from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import language_tool_python
from nltk.tokenize import word_tokenize
import nltk
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import threading
import queue

app = Flask(__name__)

# Initialize NLTK
nltk.download('punkt', quiet=True)
tool = language_tool_python.LanguageTool('en-US')
recognizer = sr.Recognizer()

# Audio recording variables
audio_queue = queue.Queue()
is_recording = False
sample_rate = 44100

# Disable irrelevant grammar rules
DISABLED_RULES = [
    'UPPERCASE_SENTENCE_START',
    'WHITESPACE_RULE',
    'EN_QUOTES',
    'COMMA_PARENTHESIS_WHITESPACE',
    'DASH_RULE'
]

def get_grammar_score(text):
    matches = [match for match in tool.check(text) 
              if match.ruleId not in DISABLED_RULES]
    words = word_tokenize(text)
    error_rate = len(matches) / max(1, len(words))
    return max(0, 100 - (error_rate * 100)), matches

def audio_callback(indata, frames, time, status):
    """This is called for each audio block from the microphone"""
    if is_recording:
        audio_queue.put(indata.copy())

def record_audio():
    """Background thread for recording audio"""
    global is_recording
    with sd.InputStream(samplerate=sample_rate,
                       channels=1,
                       dtype='int16',
                       callback=audio_callback):
        while is_recording:
            sd.sleep(100)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording
    is_recording = True
    threading.Thread(target=record_audio, daemon=True).start()
    return jsonify({'success': True})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    
    # Combine all audio blocks
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    if not audio_data:
        return jsonify({'success': False, 'error': 'No audio recorded'})
    
    # Save to temporary file
    temp_file = "temp_recording.wav"
    full_audio = np.concatenate(audio_data)
    write(temp_file, sample_rate, full_audio)
    
    # Transcribe
    try:
        with sr.AudioFile(temp_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            score, errors = get_grammar_score(text)
            
            suggestions = []
            for error in errors[:5]:
                suggestions.append({
                    'message': error.message,
                    'replacements': error.replacements[:3]
                })
            
            return jsonify({
                'success': True,
                'text': text,
                'score': score,
                'suggestions': suggestions
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    app.run(debug=True)