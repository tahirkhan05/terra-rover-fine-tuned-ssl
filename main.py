import cv2
import time
import json
import os
import threading
import numpy as np
import torch
from ultralytics import YOLO
from config.settings import settings
from services.stream_processor import StreamProcessor
from services.speech_processor import SpeechProcessor
from services.image_processor import ImageProcessor
from models.vlm_processor import VLMProcessor
from utils.parallel import ParallelProcessor
from utils.logger import logger

class PlantDiseaseDetector:
    def __init__(self, model_path='./best.pt'):
        # Load fine-tuned YOLO11s model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded fine-tuned YOLO11s model from {model_path}")
            
            # Update class names for display
            self.class_names = self.model.names
            logger.info(f"Detected {len(self.class_names)} classes")
            
            # Set detection parameters
            self.conf_threshold = 0.25
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def process_frame(self, frame):
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Process detections
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
            
            # Create annotated frame
            annotated_frame = self._annotate_frame(frame.copy(), detections)
            
            # Calculate and log processing time
            process_time = time.time() - start_time
            if len(detections) > 0:
                logger.debug(f"Detected {len(detections)} plant issues in {process_time:.3f}s")
            
            return detections, annotated_frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], frame

    def _annotate_frame(self, frame, detections):
        # Add detection boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Add specialized color coding by category
            if 'disease' in label.lower() or 'spot' in label.lower() or 'blight' in label.lower():
                color = (0, 0, 255)  # Red for diseases
            elif 'weed' in label.lower():
                color = (0, 165, 255)  # Orange for weeds
            else:
                color = (0, 255, 0)  # Green for other plants
            
            # Draw semi-transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Add label with confidence
            text = f"{label} ({conf:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), 
                         color, -1)
                         
            # Text
            cv2.putText(frame, text, 
                      (x1 + 5, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add header with detection summary
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        
        # Organize detections by category
        diseases = [d for d in detections if 'disease' in d['label'].lower() or 'spot' in d['label'].lower() or 'blight' in d['label'].lower()]
        weeds = [d for d in detections if 'weed' in d['label'].lower()]
        
        # Create summary text
        summary = f"Detected: {len(diseases)} diseases, {len(weeds)} weeds, {len(detections) - len(diseases) - len(weeds)} other"
        cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class TerraRover:
    def __init__(self):
        self.stream_processor = StreamProcessor()  # Initialize standard stream processor first
        self.plant_detector = PlantDiseaseDetector()
        self._setup_plant_detector()  # Then set up the plant detector
        self.speech_processor = SpeechProcessor()
        self.vlm_processor = VLMProcessor()
        self.image_processor = ImageProcessor()
        self.parallel_processor = ParallelProcessor(settings.MAX_WORKERS)
        self.running = False
        self.last_vlm_call = 0
        self.vlm_cooldown = 1.0  # Minimum seconds between VLM calls
        self.processing_voice = False  # Flag to prevent multiple voice queries
        self.status_message = ""
        self.status_message_timeout = 0

    def _setup_plant_detector(self):
        """Setup the plant disease detector as the object detector for the stream processor"""
        # Replace default object detector with our plant disease detector
        self.stream_processor.object_detector.detect_objects = self.plant_detector.process_frame
        logger.info("Plant disease detector configured as object detector")

    def start(self):
        logger.info("Starting Terra Rover Plant Disease Detection System")
        self.running = True
        
        # Start subsystems
        capture_thread, process_thread = self.stream_processor.start()
        self._start_status_monitor()
        
        # Configure OpenCV window properties
        cv2.namedWindow("Terra Rover", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Terra Rover", settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        
        # Print startup message with instructions
        print("\n" + "="*50)
        print("Terra Rover Plant Disease Detection System Started")
        print("="*50)
        print("Loaded plant disease detection model with classes:")
        for idx, cls_name in self.plant_detector.class_names.items():
            print(f"  - {idx}: {cls_name}")
        print("\nPress 's' to ask a question about plants in view")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        # Set status message for UI
        self.set_status_message("System ready. Press 's' to ask a question.", 5)
        
        try:
            while self.running:
                self._process_frame()
                self._handle_input()
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self._shutdown(capture_thread, process_thread)

    def _process_frame(self):
        """Handle frame processing pipeline"""
        detections, frame = self.stream_processor.get_latest_detection()
        if frame is not None:
            # Save to S3 in parallel
            self.parallel_processor.submit_task(
                self.image_processor.process_and_store_frame,
                frame
            )
            
            # Display with status message if needed
            if self.status_message and time.time() < self.status_message_timeout:
                # Add status message to the frame
                frame = self._add_status_message(frame, self.status_message)
            
            # Display
            if not self.image_processor.display_frame(frame):
                self.running = False

    def _add_status_message(self, frame, message):
        """Add status message to the frame with support for longer messages"""
        h, w, _ = frame.shape
        
        # Create a copy to avoid modifying original
        frame_with_message = frame.copy()
        
        # Create a semi-transparent overlay for the message box
        overlay = frame_with_message.copy()
        
        # Calculate text size for proper box dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Handle multi-line message display
        lines = []
        max_width = w - 100  # Maximum width for text box
        
        # Split long message into multiple lines
        words = message.split(' ')
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + ' ' + word
            test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            
            if test_size[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
                
        # Add the last line
        lines.append(current_line)
        
        # Calculate box height based on number of lines
        line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10
        box_height = (line_height * len(lines)) + 20  # padding
        
        # Find maximum line width
        max_line_width = 0
        for line in lines:
            line_width = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            max_line_width = max(max_line_width, line_width)
        
        # Draw background box
        padding = 10
        box_width = max_line_width + padding * 2
        box_x = (w - box_width) // 2  # Center horizontally
        box_y = 50  # From top
        
        cv2.rectangle(overlay, 
                    (box_x, box_y), 
                    (box_x + box_width, box_y + box_height), 
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_with_message, 0.3, 0, frame_with_message)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y_position = box_y + (i+1) * line_height
            cv2.putText(frame_with_message, line, 
                    (box_x + padding, y_position), 
                    font, font_scale, (255, 255, 255), thickness)
        
        return frame_with_message

    def set_status_message(self, message, duration=3):
        """Set a status message to display on the UI for a duration in seconds"""
        self.status_message = message
        self.status_message_timeout = time.time() + duration
        logger.debug(f"Status message set: {message[:50]}{'...' if len(message) > 50 else ''}")

    def _handle_input(self):
        """Process user input with non-blocking voice query"""
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord('q'):
            self.running = False
        elif key == ord('s') and (current_time - self.last_vlm_call) > self.vlm_cooldown and not self.processing_voice:
            self.last_vlm_call = current_time
            
            # Ensure frame queue is not empty before attempting to access frames
            if self.stream_processor.detection_queue.empty():
                print("❌ Waiting for frame to be processed...")
                self.set_status_message("Waiting for frame to be processed...", 3)
                # Wait a moment and try again
                time.sleep(0.5)
                
            # Capture the current frame before starting the thread
            detections, current_frame = self.stream_processor.get_latest_detection()
            if current_frame is None:
                print("❌ No video frame available")
                self.set_status_message("No video frame available", 3)
                return
                
            # Clone the frame to avoid any threading issues
            frame_copy = current_frame.copy()
            
            # Start voice processing with the captured frame
            self.processing_voice = True
            self.set_status_message("Voice query activated!", 2)
            print("\n🔊 Voice query activated!")
            
            # Pass the captured frame to the voice processing thread
            threading.Thread(
                target=self._process_voice_query, 
                args=(frame_copy,), 
                daemon=True
            ).start()

    def _process_voice_query(self, frame):
        """Enhanced voice query handling with better feedback and full answer display"""
        logger.info("Starting voice query processing...")
        
        try:
            # 1. Capture audio
            logger.debug("Starting audio recording...")
            self.set_status_message("Listening... Speak now", 5)
            audio = self.speech_processor.record_audio(duration=5)
            if not audio:
                logger.error("No audio data captured")
                print("❌ No audio detected. Please try again.")
                self.set_status_message("No audio detected. Try again.", 3)
                self.processing_voice = False  # Reset flag
                return
                
            # 2. Transcribe
            logger.debug("Starting speech transcription...")
            self.set_status_message("Transcribing your speech...", 3)
            question = self.speech_processor.transcribe_speech(audio)
            if not question:
                logger.error("No transcription returned")
                self.set_status_message("Couldn't understand audio. Try again.", 3)
                self.processing_voice = False  # Reset flag
                return
                
            logger.info(f"Transcribed question: {question}")
            print(f"🎙️ Your question: {question}")
            self.set_status_message(f"Question: {question}", 5)
            
            # 4. Save frame
            logger.debug("Processing and storing frame...")
            print("🖼️ Processing current frame...")
            self.set_status_message("Processing and analyzing image...", 3)
            s3_path, image_key = self.image_processor.process_and_store_frame(frame)
            if not image_key:
                logger.error("Failed to store frame in S3")
                print("❌ Failed to store image for analysis")
                self.set_status_message("Failed to process image", 3)
                self.processing_voice = False  # Reset flag
                return
                
            logger.debug(f"Frame stored at: {s3_path}")
            
            # 5. Process with VLM
            logger.debug("Invoking VLM...")
            self.set_status_message("Generating answer...", 5)
            print("🤖 Analyzing image and generating response...")
            
            # Check if VLM model ID is configured
            if not settings.VLM_MODEL_ID:
                logger.error("VLM model ID not configured")
                print("❌ VLM model not configured in .env file")
                self.set_status_message("VLM model not configured", 3)
                self.processing_voice = False  # Reset flag 
                return
                
            response = self.vlm_processor.generate_response(
                query_type='plant_disease',  # Updated to plant-specific query type
                image_key=image_key,
                question=question
            )
            
            # Display full response as status message with longer timeout
            self.set_status_message(f"Answer: {response}", 20)
            
            # Pretty print the response
            print("\n" + "="*50)
            print("✅ ANSWER:")
            print(f"{response}")
            print("="*50 + "\n")
            
            logger.info(f"VLM Response: {response}")
            
        except Exception as e:
            logger.error(f"Error in voice query processing: {str(e)}")
            print(f"❌ Error processing your query: {str(e)}")
            self.set_status_message(f"Error: {str(e)}", 3)
        finally:
            # Always reset the processing flag when done
            self.processing_voice = False

    def _start_status_monitor(self):
        """Enhanced system monitoring"""
        def monitor():
            while self.running:
                time.sleep(10)  # Reduced frequency
                stats = {
                    "fps": 0,
                    "queue": self.stream_processor.frame_queue.qsize(),
                    "detection_queue": self.stream_processor.detection_queue.qsize(),
                    "processing_time": 0
                }
                
                if self.stream_processor.processing_times:
                    stats["fps"] = 1/np.mean(self.stream_processor.processing_times)
                    stats["processing_time"] = np.mean(self.stream_processor.processing_times)
                
                logger.info(
                    "System Status | "
                    f"FPS: {stats['fps']:.1f} | "
                    f"Queue: {stats['queue']}/{self.stream_processor.max_queue_size} | "
                    f"Proc Time: {stats['processing_time']*1000:.1f}ms"
                )
                
        threading.Thread(target=monitor, daemon=True).start()

    def _shutdown(self, *threads):
        """Enhanced graceful shutdown procedure"""
        logger.info("Initiating shutdown...")
        print("\n" + "="*50)
        print("Shutting down Terra Rover Plant Disease Detection...")
        self.running = False
        
        # Stop stream processor first
        self.stream_processor.stop()
        
        # Shutdown parallel processor
        self.parallel_processor.executor.shutdown(wait=False, cancel_futures=True)
        
        # Force terminate any remaining threads
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                logger.warning(f"Terminating lingering thread: {thread.name}")
                try:
                    thread._stop()  # Force stop for stubborn threads
                except:
                    pass
        
        cv2.destroyAllWindows()
        print("Shutdown complete! Thank you for using Terra Rover Plant Disease Detection.")
        print("="*50 + "\n")
        logger.info("System shutdown complete")
        os._exit(0)  # Force exit if normal shutdown fails

if __name__ == "__main__":
    rover = TerraRover()
    rover.start()