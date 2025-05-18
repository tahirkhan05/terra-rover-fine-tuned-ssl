import logging
import sys
from config.settings import settings

def setup_logger():
    logger = logging.getLogger("terra_rover")
    logger.setLevel(settings.LOG_LEVEL)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(handler)
    return logger

logger = setup_logger()