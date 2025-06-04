import sounddevice as sd
import numpy as np
import time
import base64
from io import BytesIO
from scipy.io.wavfile import write
import pyttsx3
from services.aws_client import AWSClient
from utils.logger import logger
from config.settings import settings

class SpeechProcessor:
    def __init__(self):
        self.aws_client = AWSClient()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.recording = False
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            # Set voice (prefer female voice for agricultural assistant)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.tts_engine = None
        
        self._verify_microphone()
        logger.info("SpeechProcessor with TTS initialized successfully")
    
    def speak_text(self, text):
        """Convert text to speech"""
        if not self.tts_engine:
            logger.warning("TTS engine not available")
            return False
            
        try:
            # Clean text for better TTS
            clean_text = text.replace('\n', ' ').replace('  ', ' ').strip()
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "... continuing analysis available on screen."
            
            logger.info(f"Speaking text: {clean_text[:50]}...")
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def _verify_microphone(self):
        """Microphone verification"""
        try:
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {devices}")
            
            default_input = sd.default.device[0]
            logger.info(f"Default input device: {devices[default_input]['name']}")
            
            if not devices[default_input]['max_input_channels'] > 0:
                raise Exception("No input channels available on default device")
            
            logger.info("Testing microphone...")
            test_recording = sd.rec(int(0.5 * self.sample_rate), 
                                  samplerate=self.sample_rate,
                                  channels=self.channels,
                                  dtype=self.dtype)
            sd.wait()
            
            max_amplitude = np.abs(test_recording).max()
            logger.info(f"Test recording max amplitude: {max_amplitude}")
            
            if max_amplitude < 10:
                logger.warning("WARNING: Low microphone levels detected")
            
        except Exception as e:
            logger.error(f"Microphone verification failed: {str(e)}")

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            print(f"\n🎤 Speak now... Recording for {duration} seconds")
            
            recording = sd.rec(int(duration * self.sample_rate),
                            samplerate=self.sample_rate,
                            channels=self.channels,
                            dtype=self.dtype)
            
            for i in range(duration, 0, -1):
                print(f"Recording... {i} seconds left", end="\r")
                time.sleep(1)
            
            print("Processing audio...                ")
            sd.wait()  
            
            max_amplitude = np.abs(recording).max()
            logger.info(f"Recording max amplitude: {max_amplitude}")
            
            if max_amplitude < 10:
                print("❌ No sound detected!")
                return None
                
            audio_buffer = BytesIO()
            write(audio_buffer, self.sample_rate, recording)
            audio_buffer.seek(0)
            
            print("✅ Audio recording completed!")
            return audio_buffer.read()
        except Exception as e:
            print("❌ Audio recording failed!")
            logger.error(f"Audio recording error: {str(e)}")
            return None

    def transcribe_speech(self, audio_bytes):
        """Process audio with Amazon Lex"""
        if not audio_bytes:
            return None
            
        try:
            if not settings.LEX_BOT_ID or not settings.LEX_BOT_ALIAS_ID:
                logger.error("Lex configuration missing")
                return None
                
            print("🔄 Transcribing speech...")
            response = self.aws_client.recognize_speech(audio_bytes)
            
            if response:
                logger.info(f"Transcription: {response}")
                print(f"🎙️ Transcription: {response}")
                return response
            else:
                print("❌ No transcription returned")
                return None
        except Exception as e:
            logger.error(f"Speech transcription error: {str(e)}")
            return None
