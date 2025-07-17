import cv2
import numpy as np
import torch
from model_loader import safe_load_model
from utils.logger import logger
import time

class ObjectDetector:
    def __init__(self, model_path):
        # YOLO11 configuration
        self.model = safe_load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Performance optimization settings for YOLO11s
        self.conf_threshold = 0.25  # Lower confidence threshold for faster processing
        self.iou_threshold = 0.45   # IOU threshold for NMS
        self.max_det = 300          # Maximum detections per image
        
        # Load model in half precision if on CUDA
        if self.device == 'cuda':
            self.half = True
            self.model.half()
        else:
            self.half = False
            
        # Warmup the model
        self._warmup()
        
        logger.info(f"Object detection model (YOLO11s) loaded on {self.device} {'(half precision)' if self.half else ''}")
    
    def _warmup(self):
        """Warm up the model with a dummy inference to initialize optimizations"""
        try:
            dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
            if self.half:
                dummy_input = dummy_input.half()
                
            for _ in range(3):  # Multiple warm-up runs
                _ = self.model(dummy_input)
                
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def detect_objects(self, frame):
        try:
            start_time = time.time()
            
            # Use half precision for faster inference if GPU available
            with torch.cuda.amp.autocast(enabled=self.half):
                # Configure inference parameters for speed
                results = self.model(
                    frame, 
                    verbose=False,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    augment=False        # Disable augmentation for speed
                )
            
            detected_objects = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    
                    detected_objects.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            inference_time = (time.time() - start_time) * 1000
            if inference_time > 25:  # Log if inference takes more than 25ms
                logger.debug(f"Detection took {inference_time:.1f}ms, found {len(detected_objects)} objects")
            
            # Create a clean copy of the frame for our improved visualization
            improved_frame = frame.copy()
            
            # Draw improved bounding boxes and labels
            for obj in detected_objects:
                x1, y1, x2, y2 = obj['bbox']
                label = obj['label']
                conf = obj['confidence']
                
                # Draw a semi-transparent background for the box
                overlay = improved_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (66, 135, 245), 2)
                cv2.addWeighted(overlay, 0.4, improved_frame, 0.6, 0, improved_frame)
                
                # Create a nice text label with confidence
                text = f"{label} ({conf:.2f})"
                
                # Get text size for better placement
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(improved_frame, 
                             (x1, y1 - text_size[1] - 10), 
                             (x1 + text_size[0] + 10, y1), 
                             (66, 135, 245), -1)
                
                # Draw text with improved visibility
                cv2.putText(improved_frame, text, 
                          (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return detected_objects, improved_frame
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            return [], frame
