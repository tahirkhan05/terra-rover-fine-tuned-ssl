import cv2
import os
import time
import numpy as np
from config.settings import settings
from utils.logger import logger

class ImageProcessor:
    def __init__(self):
        os.makedirs(settings.LOCAL_SAVE_PATH, exist_ok=True)
        self.ui_font = cv2.FONT_HERSHEY_SIMPLEX
        self.ui_scale = 0.6
        self.ui_color = (255, 255, 255)
        self.ui_thickness = 2
    
    def save_frame_locally(self, frame):
        try:
            timestamp = int(time.time() * 1000)  # Milliseconds for unique filename
            filename = f"{settings.LOCAL_SAVE_PATH}/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            logger.error(f"Frame save error: {str(e)}")
            return None
    
    def display_frame(self, frame, window_name='Terra Rover'):
        # Create status bar at the bottom of the frame
        h, w, _ = frame.shape
        frame_with_ui = self._add_ui_elements(frame)
        
        # Set window properties for a better UI experience
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
        cv2.imshow(window_name, frame_with_ui)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def _add_ui_elements(self, frame):
        """Add UI elements to the frame for better visualization"""
        h, w, _ = frame.shape
        
        ui_frame = frame.copy()
        
        # Add semi-transparent status bar at the bottom
        status_bar_height = 40
        overlay = ui_frame.copy()
        cv2.rectangle(overlay, (0, h - status_bar_height), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0, ui_frame)
        
        help_text = "`s` Voice input | `t` Type input | `l` SSL bg learn | `p` SSL frame processing | `r` SSL status | `q` Quit "
        cv2.putText(ui_frame, help_text, 
                  (10, h - 12), 
                  cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return ui_frame
    
    def process_and_store_frame(self, frame):
        """Process and store captured frame with verification"""
        from services.aws_client import AWSClient
        import cv2
        import time
        
        # Save locally
        local_path = self.save_frame_locally(frame)
        if not local_path:
            return None, None
        
        # Compress for S3 upload
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to compress frame")
            return None, None
            
        # Upload to S3
        aws_client = AWSClient()
        timestamp = int(time.time() * 1000)
        key = f"frames/frame_{timestamp}.jpg"
        
        try:
            aws_client.s3.put_object(
                Bucket=settings.S3_BUCKET,
                Key=key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )
            
            # Verify upload
            aws_client.s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
            logger.debug(f"Successfully uploaded frame to s3://{settings.S3_BUCKET}/{key}")
            return f"s3://{settings.S3_BUCKET}/{key}", key
        except Exception as e:
            logger.error(f"Failed to upload frame to S3: {str(e)}")
            return None, None