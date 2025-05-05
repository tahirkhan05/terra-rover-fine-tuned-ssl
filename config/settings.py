import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # RTSP Stream Configuration
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.1.100:8554/live.sdp")
    FPS = int(os.getenv("FPS", 60)) 
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1280)) 
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET", "terra-rover-bucket")
    
    # Model Configuration
    OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "yolo11s.pt")
    VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Speech Configuration
    LEX_BOT_ID = os.getenv("LEX_BOT_ID", "TerraRoverBot")
    LEX_BOT_ALIAS_ID = os.getenv("LEX_BOT_ALIAS_ID", "Prod")
    LEX_LOCALE_ID = os.getenv("LEX_LOCALE_ID", "en_US")
    
    # System Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOCAL_SAVE_PATH = os.getenv("LOCAL_SAVE_PATH", "data/captured_frames")

settings = Settings()
