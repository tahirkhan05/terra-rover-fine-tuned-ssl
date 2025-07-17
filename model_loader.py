import torch
from ultralytics import YOLO
from utils.logger import logger
import os

def safe_load_model(model_path):
    """Safely load YOLO model with YOLO11s compatibility"""
    try:
        # Check if file exists locally, download if not
        if not os.path.exists(model_path) and model_path == "yolo11s.pt":
            logger.info("YOLO11s model not found locally, downloading...")
            
            # Let Ultralytics handle the download
            model = YOLO("yolo11s.pt")
            logger.info(f"YOLO11s model downloaded successfully")
            return model
            
        # Standard load for existing files
        return YOLO(model_path)
    except Exception as e:
        logger.warning(f"Standard model load failed, attempting workaround: {str(e)}")
        
        # Enhanced workaround for newer PyTorch and Ultralytics versions
        try:
            # Try alternative loading method with explicit task
            model = YOLO(model_path, task='detect')
            logger.info("Model loaded with explicit task specification")
            return model
        except Exception as e2:
            logger.warning(f"Alternative loading failed: {str(e2)}")
            
            # Attempt compatibility mode workaround
            import torch.serialization
            from ultralytics.nn.tasks import DetectionModel, attempt_load_weights
            
            # Add necessary classes to safe globals
            torch.serialization.add_safe_globals([DetectionModel, attempt_load_weights])
            
            # Try loading again with compatibility mode
            logger.info("Attempting compatibility mode loading...")
            return YOLO(model_path)
