import sounddevice as sd
import numpy as np
import time
import base64
from io import BytesIO
from scipy.io.wavfile import write
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
        self._verify_microphone()
        logger.info("SpeechProcessor initialized successfully")
    
    def _verify_microphone(self):
        """Microphone verification"""
        try:
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {devices}")
            
            default_input = sd.default.device[0]
            logger.info(f"Default input device: {devices[default_input]['name']}")
            
            if not devices[default_input]['max_input_channels'] > 0:
                raise Exception("No input channels available on default device")
            
            # Test recording with logging
            logger.info("Testing microphone with short recording...")
            test_recording = sd.rec(int(0.5 * self.sample_rate), 
                                  samplerate=self.sample_rate,
                                  channels=self.channels,
                                  dtype=self.dtype)
            sd.wait()
            
            max_amplitude = np.abs(test_recording).max()
            logger.info(f"Test recording max amplitude: {max_amplitude}")
            
            if max_amplitude < 10:
                logger.warning("WARNING: Microphone test recording detected very low sound levels")
                logger.warning("Please check your microphone connection and volume settings")
            
        except Exception as e:
            logger.error(f"Microphone verification failed: {str(e)}")
            logger.error("Voice commands will not work without a functioning microphone")

    def record_audio(self, duration=5):
        """Record audio from microphone with enhanced feedback"""
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            print("\n🎤 Speak now... Recording for {duration} seconds".format(duration=duration))
            
            # Flash the console to make it obvious recording is happening
            for _ in range(3):
                print("Recording...", end="\r")
                time.sleep(0.2)
                print("           ", end="\r")
                time.sleep(0.2)
            
            recording = sd.rec(int(duration * self.sample_rate),
                            samplerate=self.sample_rate,
                            channels=self.channels,
                            dtype=self.dtype)
            
            # Visual countdown
            for i in range(duration, 0, -1):
                print(f"Recording... {i} seconds left", end="\r")
                time.sleep(1)
            
            print("Processing audio...                ")
            sd.wait()  
            
            # Check if audio was captured
            max_amplitude = np.abs(recording).max()
            logger.info(f"Recording max amplitude: {max_amplitude}")
            
            if max_amplitude < 10:
                print("❌ No sound detected! Check your microphone.")
                logger.warning(f"No sound detected in recording (max amplitude: {max_amplitude})")
                return None
                
            # Convert to WAV format
            audio_buffer = BytesIO()
            write(audio_buffer, self.sample_rate, recording)
            audio_buffer.seek(0)
            
            print("✅ Audio recording completed!")
            logger.info("Audio recording completed successfully")
            return audio_buffer.read()
        except Exception as e:
            print("❌ Audio recording failed!")
            logger.error(f"Audio recording error: {str(e)}")
            return None

    def transcribe_speech(self, audio_bytes):
        """Process audio with Amazon Lex and return transcription"""
        if not audio_bytes:
            logger.error("No audio data provided for transcription")
            return None
            
        try:
            # Check if Lex configuration is set
            if not settings.LEX_BOT_ID or not settings.LEX_BOT_ALIAS_ID:
                logger.error("Lex configuration is missing! Check your .env file")
                print("❌ Speech recognition not configured. Check your .env file.")
                return None
                
            print("🔄 Transcribing speech...")
            # Using the AWS client's recognize_speech method
            response = self.aws_client.recognize_speech(audio_bytes)
            
            if response:
                # Check if response is a base64/encoded string and try to decode it
                if isinstance(response, str) and (response.startswith("H4s") or 
                                                "%" in response or 
                                                "+" in response or
                                                response.startswith("data:") or
                                                "=" in response):
                    logger.warning("Received possible encoded response, attempting to decode")
                    try:
                        # Try to decode if it's base64
                        decoded = None
                        try:
                            # Try standard base64 decode first
                            decoded = base64.b64decode(response).decode('utf-8')
                        except:
                            # Try url-safe base64
                            try:
                                decoded = base64.urlsafe_b64decode(response).decode('utf-8')
                            except:
                                pass
                        
                        if decoded:
                            logger.info(f"Successfully decoded response: {decoded}")
                            response = decoded
                    except Exception as decode_error:
                        logger.warning(f"Failed to decode response: {decode_error}")
                
                logger.info(f"Transcription: {response}")
                print(f"🎙️ Transcription: {response}")
            else:
                logger.warning("No transcription returned from Lex")
                print("❌ No transcription returned. Try speaking clearly.")
            
            return response
        except Exception as e:
            logger.error(f"Speech transcription error: {str(e)}")
            print(f"❌ Speech transcription failed: {str(e)}")
            return None