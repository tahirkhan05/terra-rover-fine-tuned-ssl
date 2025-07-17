import os
import sys
import threading
import time
from typing import Dict, Optional
import logging

# Add the models directory to path for SSL framework import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from ssl_framework import TerraRoverSSLIntegration, ContinuousLearningManager
from config.settings import settings
from utils.logger import logger

class SSLService:
    """Service for managing SSL integration in Terra Rover"""
    
    def __init__(self, config_path: str = "config/ssl_config.json"):
        self.config_path = config_path
        self.ssl_integration = None
        self.learning_thread = None
        self.learning_active = False
        self.frame_processing_enabled = True
        
        # Performance metrics
        self.metrics = {
            'total_frames_processed': 0,
            'ssl_learning_sessions': 0,
            'last_ssl_loss': 0.0,
            'average_confidence': 0.0
        }
        
        # Initialize SSL system
        self._initialize_ssl()
        
        logger.info("SSL Service initialized")
    
    def _initialize_ssl(self):
        """Initialize the SSL integration system"""
        try:
            self.ssl_integration = TerraRoverSSLIntegration(self.config_path)
            logger.info("SSL integration system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SSL system: {str(e)}")
            self.ssl_integration = None
    
    def start_background_learning(self):
        """Start background SSL learning"""
        if not self.ssl_integration:
            logger.warning("SSL integration not available, cannot start background learning")
            return False
        
        if self.learning_active:
            logger.info("Background learning already active")
            return True
        
        self.learning_active = True
        self.learning_thread = threading.Thread(
            target=self._background_learning_loop,
            daemon=True,
            name="SSL_Background_Learning"
        )
        self.learning_thread.start()
        logger.info("Background SSL learning started")
        return True
    
    def stop_background_learning(self):
        """Stop background SSL learning"""
        if self.learning_active:
            self.learning_active = False
            if self.learning_thread and self.learning_thread.is_alive():
                self.learning_thread.join(timeout=5.0)
            logger.info("Background SSL learning stopped")
    
    def _background_learning_loop(self):
        """Background learning loop"""
        while self.learning_active:
            try:
                if self.ssl_integration:
                    # Get current status
                    status = self.ssl_integration.get_system_status()
                    buffer_size = status['ssl_stats']['current_buffer_size']
                    
                    # Check if learning should be triggered based on buffer size
                    learning_threshold = self.ssl_integration.config.get('learning_threshold', 100)
                    
                    if buffer_size >= learning_threshold:
                        logger.info(f"Triggering SSL learning session (buffer: {buffer_size})")
                        self.metrics['ssl_learning_sessions'] += 1
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in background learning loop: {str(e)}")
                time.sleep(10)  # Wait before retrying
    
    def process_frame_with_ssl(self, frame_path: str) -> Dict:
        """Process a frame with SSL integration"""
        if not self.ssl_integration or not self.frame_processing_enabled:
            return {'confidence': 0.0, 'detections': [], 'ssl_enabled': False}
        
        try:
            # Process frame with SSL
            detection_results = self.ssl_integration.process_frame(frame_path)
            
            # Update metrics
            self.metrics['total_frames_processed'] += 1
            
            # Update average confidence
            current_conf = detection_results.get('confidence', 0.0)
            total_frames = self.metrics['total_frames_processed']
            self.metrics['average_confidence'] = (
                (self.metrics['average_confidence'] * (total_frames - 1) + current_conf) / total_frames
            )
            
            # Add SSL metadata
            detection_results['ssl_enabled'] = True
            detection_results['ssl_buffer_size'] = len(self.ssl_integration.ssl_manager.frame_buffer)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error processing frame with SSL: {str(e)}")
            return {'confidence': 0.0, 'detections': [], 'ssl_enabled': False, 'error': str(e)}
    
    def get_ssl_status(self) -> Dict:
        """Get comprehensive SSL status"""
        if not self.ssl_integration:
            return {
                'ssl_available': False,
                'learning_active': False,
                'error': 'SSL integration not initialized'
            }
        
        try:
            system_status = self.ssl_integration.get_system_status()
            
            return {
                'ssl_available': True,
                'learning_active': self.learning_active,
                'ssl_stats': system_status['ssl_stats'],
                'yolo_model_loaded': system_status['yolo_model_loaded'],
                'config': system_status['config'],
                'performance_metrics': self.metrics,
                'frame_processing_enabled': self.frame_processing_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting SSL status: {str(e)}")
            return {
                'ssl_available': False,
                'learning_active': False,
                'error': str(e)
            }
    
    def toggle_frame_processing(self):
        """Toggle SSL frame processing on/off"""
        self.frame_processing_enabled = not self.frame_processing_enabled
        status = "enabled" if self.frame_processing_enabled else "disabled"
        logger.info(f"SSL frame processing {status}")
        return self.frame_processing_enabled
    
    def get_learning_statistics(self) -> Dict:
        """Get detailed learning statistics"""
        if not self.ssl_integration:
            return {'error': 'SSL integration not available'}
        
        try:
            ssl_stats = self.ssl_integration.ssl_manager.get_learning_statistics()
            
            return {
                'ssl_learning_stats': ssl_stats,
                'service_metrics': self.metrics,
                'learning_active': self.learning_active,
                'frame_processing_enabled': self.frame_processing_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {str(e)}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown SSL service"""
        logger.info("Shutting down SSL service...")
        self.stop_background_learning()
        
        if self.ssl_integration:
            # Perform any cleanup if needed
            pass
        
        logger.info("SSL service shutdown complete")
