import cv2
import time
import json
import os
import threading
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from ultralytics import YOLO
from config.settings import settings
from services.stream_processor import StreamProcessor
from services.speech_processor import SpeechProcessor
from services.image_processor import ImageProcessor
from models.vlm_processor import VLMProcessor
from utils.parallel import ParallelProcessor
from utils.logger import logger

class TextOverlay:
    def __init__(self):
        self.active = False
        self.text = ""
        self.cursor_pos = 0
        self.blink_state = True
        self.last_blink = time.time()
        
    def activate(self):
        self.active = True
        self.text = ""
        self.cursor_pos = 0
        
    def deactivate(self):
        self.active = False
        self.text = ""
        
    def handle_key(self, key):
        if not self.active:
            return None
            
        if key == 13:  # Enter
            result = self.text.strip()
            self.deactivate()
            return result
        elif key == 27:  # Escape
            self.deactivate()
            return None
        elif key == 8:  # Backspace
            if self.cursor_pos > 0:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
        elif 32 <= key <= 126:  # Printable characters
            char = chr(key)
            self.text = self.text[:self.cursor_pos] + char + self.text[self.cursor_pos:]
            self.cursor_pos += 1
            
        return "continue"
        
    def draw_on_frame(self, frame):
        if not self.active:
            return frame
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Input box
        box_width = w - 100
        box_height = 120
        box_x = 50
        box_y = h - box_height - 50
        
        # Semi-transparent background
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (45, 80, 22), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "Agricultural Analysis Query", (box_x + 10, box_y + 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Type your question about crops, diseases, pests, or farming:", 
                   (box_x + 10, box_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Input field background
        input_x = box_x + 10
        input_y = box_y + 60
        input_width = box_width - 20
        input_height = 30
        cv2.rectangle(frame, (input_x, input_y), (input_x + input_width, input_y + input_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (input_x, input_y), (input_x + input_width, input_y + input_height), (0, 0, 0), 1)
        
        # Text - FIXED: Increased font size from 0.6 to 0.8 for better readability
        if self.text:
            cv2.putText(frame, self.text, (input_x + 5, input_y + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Cursor
        if time.time() - self.last_blink > 0.5:
            self.blink_state = not self.blink_state
            self.last_blink = time.time()
            
        if self.blink_state:
            # FIXED: Updated cursor position calculation for new font size
            cursor_x = input_x + 5 + cv2.getTextSize(self.text[:self.cursor_pos], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            cv2.line(frame, (cursor_x, input_y + 5), (cursor_x, input_y + 25), (0, 0, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press ENTER to submit | ESC to cancel", 
                   (box_x + 10, box_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        return frame

class PlantDiseaseDetector:
    def __init__(self, model_path='./best.pt'):
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded fine-tuned YOLO11s model from {model_path}")
            
            self.class_names = self.model.names
            logger.info(f"Detected {len(self.class_names)} classes")
            
            self.conf_threshold = 0.25
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def process_frame(self, frame):
        try:
            start_time = time.time()
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.class_names[class_id]
                    
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            annotated_frame = self._annotate_frame(frame.copy(), detections)
            
            process_time = time.time() - start_time
            if len(detections) > 0:
                logger.debug(f"Detected {len(detections)} plant issues in {process_time:.3f}s")
            
            return detections, annotated_frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], frame

    def _annotate_frame(self, frame, detections):
        # Enhanced font settings for better readability
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 2
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Color coding by agricultural category
            if 'disease' in label.lower() or 'spot' in label.lower() or 'blight' in label.lower():
                color = (0, 0, 255)  # Red for diseases
            elif 'weed' in label.lower():
                color = (0, 165, 255)  # Orange for weeds
            elif 'pest' in label.lower() or 'insect' in label.lower():
                color = (255, 0, 255)  # Magenta for pests
            else:
                color = (0, 255, 0)  # Green for healthy plants
            
            # Draw thicker, more visible box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Enhanced label with better readability
            text = f"{label} ({conf:.2f})"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Larger background box for text
            padding = 8
            cv2.rectangle(frame, 
                         (x1, y1 - text_size[1] - padding*2), 
                         (x1 + text_size[0] + padding*2, y1), 
                         color, -1)
                         
            # White text with shadow effect for better visibility
            cv2.putText(frame, text, 
                      (x1 + padding, y1 - padding), 
                      font, font_scale, (0, 0, 0), font_thickness + 1)  # Shadow
            cv2.putText(frame, text, 
                      (x1 + padding, y1 - padding), 
                      font, font_scale, (255, 255, 255), font_thickness)  # Main text
        
        # Enhanced header with larger, more readable font
        header_height = 50
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (0, 0, 0), -1)
        
        # Categorize detections
        diseases = [d for d in detections if 'disease' in d['label'].lower() or 'spot' in d['label'].lower()]
        weeds = [d for d in detections if 'weed' in d['label'].lower()]
        pests = [d for d in detections if 'pest' in d['label'].lower() or 'insect' in d['label'].lower()]
        
        summary = f"Agricultural Analysis | Diseases: {len(diseases)} | Weeds: {len(weeds)} | Pests: {len(pests)}"
        cv2.putText(frame, summary, (15, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class TerraRover:
    def __init__(self):
        self.stream_processor = StreamProcessor()
        self.plant_detector = PlantDiseaseDetector()
        self._setup_plant_detector()
        self.speech_processor = SpeechProcessor()
        self.vlm_processor = VLMProcessor()
        self.image_processor = ImageProcessor()
        self.parallel_processor = ParallelProcessor(settings.MAX_WORKERS)
        self.running = False
        self.last_vlm_call = 0
        self.vlm_cooldown = 1.0
        self.processing_voice = False
        self.status_message = ""
        self.status_message_timeout = 0
        
        # Add frame buffer to store latest valid frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.text_overlay = TextOverlay()

    def _setup_plant_detector(self):
        self.stream_processor.object_detector.detect_objects = self.plant_detector.process_frame
        logger.info("Plant disease detector configured")

    def start(self):
        logger.info("Starting Terra Rover Agricultural Analysis System")
        self.running = True
        
        capture_thread, process_thread = self.stream_processor.start()
        self._start_status_monitor()
        
        cv2.namedWindow("Terra Rover Agricultural Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Terra Rover Agricultural Analysis", settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        
        print("\n" + "="*60)
        print("🌾 Terra Rover Agricultural Analysis System Started 🌾")
        print("="*60)
        print("Loaded agricultural detection model with classes:")
        for idx, cls_name in self.plant_detector.class_names.items():
            print(f"  - {idx}: {cls_name}")
        print("\nControls:")
        print("  's' - Voice question about agriculture")
        print("  't' - Type question about agriculture") 
        print("  'q' - Quit system")
        print("="*60 + "\n")
        
        self.set_status_message("Agricultural Analysis Ready | Press 't' for text input or 's' for voice", 8)
        
        try:
            while self.running:
                self._process_frame()
                self._handle_input()
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self._shutdown(capture_thread, process_thread)

    def _process_frame(self):
        detections, frame = self.stream_processor.get_latest_detection()
        if frame is not None:
            # Store the latest valid frame
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            self.parallel_processor.submit_task(
                self.image_processor.process_and_store_frame,
                frame
            )
            
            if self.status_message and time.time() < self.status_message_timeout:
                frame = self._add_status_message(frame, self.status_message)
            
            # Add text overlay if active
            frame = self.text_overlay.draw_on_frame(frame)
            
            if not self.image_processor.display_frame(frame, "Terra Rover Agricultural Analysis"):
                self.running = False

    def _get_current_frame(self):
        """Get the latest available frame with retry logic"""
        max_retries = 10
        retry_delay = 0.01  # 10ms delay
        
        for attempt in range(max_retries):
            # First try to get from stream processor
            detections, frame = self.stream_processor.get_latest_detection()
            if frame is not None:
                return frame
            
            # Then try from our buffer
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
            
            # Wait a bit and try again
            time.sleep(retry_delay)
        
        return None

    def _add_status_message(self, frame, message):
        h, w, _ = frame.shape
        frame_with_message = frame.copy()
        overlay = frame_with_message.copy()
        
        # Enhanced font for better readability
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        
        lines = []
        max_width = w - 100
        
        words = message.split(' ')
        current_line = words[0] if words else ""
        
        for word in words[1:]:
            test_line = current_line + ' ' + word
            test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            
            if test_size[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
                
        if current_line:
            lines.append(current_line)
        
        line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 15
        box_height = (line_height * len(lines)) + 30
        
        max_line_width = 0
        for line in lines:
            line_width = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            max_line_width = max(max_line_width, line_width)
        
        padding = 15
        box_width = max_line_width + padding * 2
        box_x = (w - box_width) // 2
        box_y = 60
        
        # Semi-transparent green background for agricultural theme
        cv2.rectangle(overlay, 
                    (box_x, box_y), 
                    (box_x + box_width, box_y + box_height), 
                    (45, 80, 22), -1)  # Dark green
        cv2.addWeighted(overlay, 0.8, frame_with_message, 0.2, 0, frame_with_message)
        
        # Add border
        cv2.rectangle(frame_with_message,
                     (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (0, 255, 0), 2)
        
        # Draw text with shadow for better visibility
        for i, line in enumerate(lines):
            y_position = box_y + (i+1) * line_height
            # Shadow
            cv2.putText(frame_with_message, line, 
                    (box_x + padding + 1, y_position + 1), 
                    font, font_scale, (0, 0, 0), thickness + 1)
            # Main text
            cv2.putText(frame_with_message, line, 
                    (box_x + padding, y_position), 
                    font, font_scale, (255, 255, 255), thickness)
        
        return frame_with_message

    def set_status_message(self, message, duration=3):
        self.status_message = message
        self.status_message_timeout = time.time() + duration
        logger.debug(f"Status message set: {message[:50]}{'...' if len(message) > 50 else ''}")

    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        # Handle text overlay input first
        if self.text_overlay.active:
            result = self.text_overlay.handle_key(key)
            if result and result != "continue":
                # User submitted text
                current_frame = self._get_current_frame()
                if current_frame is not None:
                    print(f"\n📝 Your agricultural question: {result}")
                    self.set_status_message(f"Analyzing: {result}", 3)
                    self.last_vlm_call = time.time()
                    self.processing_voice = True
                    
                    # FIXED: Enable TTS for text input by setting use_tts=True
                    threading.Thread(
                        target=self._process_agricultural_query, 
                        args=(current_frame.copy(), result, True), 
                        daemon=True
                    ).start()
                else:
                    self.set_status_message("No video frame available", 3)
            return
            
        # Normal controls when text overlay not active
        if key == ord('q'):
            self.running = False
        elif key == ord('t') and (current_time - self.last_vlm_call) > self.vlm_cooldown and not self.processing_voice:
            # Activate text overlay
            self.text_overlay.activate()
        elif key == ord('s') and (current_time - self.last_vlm_call) > self.vlm_cooldown and not self.processing_voice:
            # Voice input mode
            self._handle_voice_query()

    def _handle_voice_query(self):
        """Handle voice-based agricultural query"""
        # Get current frame with retry logic
        current_frame = self._get_current_frame()
        if current_frame is None:
            self.set_status_message("No video frame available for analysis - please wait", 3)
            logger.warning("No frame available for voice query")
            return
            
        frame_copy = current_frame.copy()
        self.last_vlm_call = time.time()
        self.processing_voice = True
        self.set_status_message("Voice query activated! Speak your agricultural question", 5)
        print("\n🔊 Voice agricultural query activated!")
        
        threading.Thread(
            target=self._process_voice_query, 
            args=(frame_copy,), 
            daemon=True
        ).start()

    def _process_voice_query(self, frame):
        """Process voice query with agricultural focus"""
        try:
            # Record and transcribe
            self.set_status_message("🎤 Listening for your agricultural question...", 5)
            audio = self.speech_processor.record_audio(duration=5)
            if not audio:
                self.set_status_message("No audio detected. Try again.", 3)
                self.processing_voice = False
                return
                
            self.set_status_message("Transcribing your agricultural question...", 3)
            question = self.speech_processor.transcribe_speech(audio)
            if not question:
                self.set_status_message("Couldn't understand audio. Try again.", 3)
                self.processing_voice = False
                return
                
            print(f"🎙️ Your agricultural question: {question}")
            
            # Process with TTS enabled
            self._process_agricultural_query(frame, question, use_tts=True)
            
        except Exception as e:
            logger.error(f"Voice query error: {e}")
            self.set_status_message(f"Voice query error: {str(e)}", 3)
            self.processing_voice = False

    def _process_agricultural_query(self, frame, question, use_tts=False):
        """Process agricultural query with enhanced VLM prompts"""
        try:
            self.set_status_message("Processing agricultural analysis...", 5)
            print("🌱 Analyzing agricultural conditions...")
            
            # Store frame
            s3_path, image_key = self.image_processor.process_and_store_frame(frame)
            if not image_key:
                error_msg = "Failed to process image for agricultural analysis"
                self.set_status_message(error_msg, 3)
                self.processing_voice = False
                return
                
            self.set_status_message("Generating agricultural insights...", 5)
            print("🤖 AI analyzing crop conditions and generating response...")
            
            # Use agriculture-specific query type
            query_type = 'plant_disease'
            if 'crop' in question.lower() or 'yield' in question.lower() or 'harvest' in question.lower():
                query_type = 'crop_analysis'
            
            response = self.vlm_processor.generate_response(
                query_type=query_type,
                image_key=image_key,
                question=question
            )
            
            # Display response
            self.set_status_message(f"Agricultural Analysis: {response}", 25)
            
            print("\n" + "="*60)
            print("🌾 AGRICULTURAL ANALYSIS RESULTS:")
            print(f"Question: {question}")
            print(f"Answer: {response}")
            print("="*60 + "\n")
            
            # Text-to-speech for both voice and text queries
            if use_tts:
                print("🔊 Speaking agricultural analysis...")
                threading.Thread(
                    target=self.speech_processor.speak_text,
                    args=(response,),
                    daemon=True
                ).start()
            
            logger.info(f"Agricultural VLM Response: {response}")
            
        except Exception as e:
            logger.error(f"Agricultural query processing error: {e}")
            error_msg = f"Agricultural analysis error: {str(e)}"
            self.set_status_message(error_msg, 5)
            print(f"❌ {error_msg}")
        finally:
            self.processing_voice = False

    def _start_status_monitor(self):
        def monitor():
            while self.running:
                time.sleep(10)
                stats = {
                    "fps": 0,
                    "queue": self.stream_processor.frame_queue.qsize(),
                    "detection_queue": self.stream_processor.detection_queue.qsize(),
                }
                
                if self.stream_processor.processing_times:
                    stats["fps"] = 1/np.mean(self.stream_processor.processing_times)
                
                logger.info(
                    "Agricultural System Status | "
                    f"FPS: {stats['fps']:.1f} | "
                    f"Queue: {stats['queue']}/{self.stream_processor.max_queue_size}"
                )
                
        threading.Thread(target=monitor, daemon=True).start()

    def _shutdown(self, *threads):
        logger.info("Shutting down Terra Rover Agricultural Analysis...")
        print("\n" + "="*60)
        print("🌾 Shutting down Terra Rover Agricultural Analysis...")
        self.running = False
        
        self.stream_processor.stop()
        self.parallel_processor.executor.shutdown(wait=False, cancel_futures=True)
        
        try:
            pass  
        except:
            pass
        
        cv2.destroyAllWindows()
        print("Agricultural analysis system shutdown complete! 🌱")
        print("="*60 + "\n")
        logger.info("Agricultural system shutdown complete")
        os._exit(0)

if __name__ == "__main__":
    rover = TerraRover()
    rover.start()
