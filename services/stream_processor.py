import cv2
import threading
import time
import numpy as np
from queue import Queue, LifoQueue
from config.settings import settings
from models.object_detection import ObjectDetector
from utils.logger import logger

class StreamProcessor:
    def __init__(self):
        self.object_detector = ObjectDetector(settings.OBJECT_DETECTION_MODEL)
        self.detection_queue = LifoQueue(maxsize=10)  # Queue size for smoother output
        self.stop_event = threading.Event()
        self.processing_times = []
        self.max_queue_size = 60 
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        self.processing_interval = 0.01 
        self.skip_frames = 0  # Dynamic frame skipping counter
        self.frame_count = 0
        self.last_fps_calc = time.time()
        self.current_fps = 0
        self.target_fps = 60  # Target processing FPS
        
        # RTSP optimization parameters
        self.cap_params = {
            cv2.CAP_PROP_BUFFERSIZE: 1,  # Keep buffer minimal to reduce latency
            cv2.CAP_PROP_FPS: settings.FPS,
            cv2.CAP_PROP_FRAME_WIDTH: settings.FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: settings.FRAME_HEIGHT,
            cv2.CAP_PROP_HW_ACCELERATION: cv2.VIDEO_ACCELERATION_ANY
        }

    def capture_frames(self):
        """Optimized RTSP capture with hardware acceleration and low latency"""
        # Try different backends to find the most efficient one
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_DSHOW]
        
        cap = None
        for backend in backends:
            try:
                logger.info(f"Trying capture backend: {backend}")
                cap = cv2.VideoCapture(settings.RTSP_URL, backend)
                if cap.isOpened():
                    logger.info(f"Successfully opened stream with backend: {backend}")
                    break
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {str(e)}")
        
        # Fall back to default if specific backends fail
        if cap is None or not cap.isOpened():
            logger.warning("Specific backends failed, trying default")
            cap = cv2.VideoCapture(settings.RTSP_URL)
        
        # Set optimized parameters
        for prop, value in self.cap_params.items():
            cap.set(prop, value)
        
        # Add connection timeout and retry logic
        retries = 5
        for attempt in range(retries):
            if cap.isOpened():
                break
            logger.warning(f"RTSP connection attempt {attempt + 1} failed")
            time.sleep(1)
        
        if not cap.isOpened():
            logger.error(f"RTSP stream failed to open after {retries} attempts")
            self.stop_event.set()
            return

        logger.info("RTSP capture started")
        
        # Record frame reception timestamps for latency monitoring
        prev_frame_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                ret, frame = cap.read()
                current_time = time.time()
                
                if not ret:
                    logger.warning("Frame capture failed - retrying")
                    time.sleep(0.05)
                    continue
                
                # Calculate and log latency occasionally
                latency = (current_time - prev_frame_time) * 1000  # ms
                if self.frame_count % 100 == 0:
                    logger.debug(f"Frame capture latency: {latency:.1f}ms")
                
                prev_frame_time = current_time
                self.frame_count += 1
                
                # Avoid queue bloat by dropping frames if necessary
                if self.frame_queue.qsize() > self.max_queue_size * 0.9:
                    continue
                    
                if self.frame_queue.full():
                    # Keep queue fresh by removing oldest frame
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                        
                self.frame_queue.put((frame, current_time))
                
                # Calculate FPS every second
                if current_time - self.last_fps_calc >= 1.0:
                    self.current_fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_calc = current_time
                    logger.debug(f"Capture FPS: {self.current_fps}")
                
            except Exception as e:
                logger.error(f"Capture error: {str(e)}")
                time.sleep(0.1)
        
        cap.release()
        logger.info("RTSP capture stopped")

    def process_detections(self):
        """Optimized detection pipeline with adaptive processing"""
        frame_skip = 0  # Initialize frame skip counter
        proc_times = []  # Track recent processing times
        
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get()
                    
                    # Dynamic frame skipping based on processing load
                    if frame_skip > 0:
                        frame_skip -= 1
                        continue
                    
                    start = time.time()
                    
                    # Skip processing if queue is backing up significantly
                    queue_load = self.frame_queue.qsize() / self.max_queue_size
                    if queue_load > 0.8:
                        # Heavy load - increase frame skipping
                        frame_skip = max(2, int(queue_load * 10))
                        logger.debug(f"Heavy load detected ({queue_load:.2f}), skipping {frame_skip} frames")
                        continue
                    
                    # Process the frame
                    detections, annotated_frame = self.object_detector.detect_objects(frame)
                    
                    # Calculate total latency from capture to completion
                    total_latency = (time.time() - timestamp) * 1000  # ms
                    if len(self.processing_times) % 50 == 0:
                        logger.debug(f"Total processing latency: {total_latency:.1f}ms")
                    
                    # Store processing time metrics
                    proc_time = time.time() - start
                    proc_times.append(proc_time)
                    if len(proc_times) > 30:
                        proc_times.pop(0)
                    
                    # Calculate mean processing time and adjust frame skipping dynamically
                    if len(proc_times) >= 10:
                        mean_proc_time = np.mean(proc_times)
                        target_time = 1.0 / self.target_fps
                        
                        # If processing is too slow, increase frame skipping
                        if mean_proc_time > target_time:
                            frame_skip = max(0, int((mean_proc_time / target_time) - 1))
                    
                    # Store metrics
                    if proc_time < 1.0:  # Ignore outliers
                        self.processing_times.append(proc_time)
                        if len(self.processing_times) > 100:
                            self.processing_times.pop(0)
                    
                    # Update detection queue with most recent result
                    if self.detection_queue.full():
                        self.detection_queue.get_nowait()
                    self.detection_queue.put((detections, annotated_frame))
                    
                else:
                    # No frames to process, short sleep
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Detection error: {str(e)}")
                time.sleep(0.01)
    
    def start(self):
        # Set thread priorities if on Linux
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_detections, daemon=True)
        
        # Start threads
        capture_thread.start()
        process_thread.start()
        
        # Report initial state
        logger.info(f"Stream processor started with target {self.target_fps} FPS")
        
        return capture_thread, process_thread
    
    def stop(self):
        self.stop_event.set()
        logger.info("Stream processor stopping...")
    
    def get_latest_detection(self):
        if not self.detection_queue.empty():
            return self.detection_queue.get()
        return None, None