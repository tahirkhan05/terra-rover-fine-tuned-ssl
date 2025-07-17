import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import asyncio
import logging
from datetime import datetime
import sqlite3
import pickle
from ultralytics import YOLO
from sklearn.cluster import KMeans
import torchvision.models as models

class SSLDataset(Dataset):
    """Self-Supervised Learning Dataset for Terra Rover"""
    
    def __init__(self, image_paths: List[str], ssl_method: str = 'simclr', 
                 augmentation_strength: float = 0.8):
        self.image_paths = image_paths
        self.ssl_method = ssl_method
        self.augmentation_strength = augmentation_strength
        
        # Define strong augmentations for SSL
        self.strong_augment = A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.ToGray(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.weak_augment = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.ssl_method == 'simclr':
            # Generate two different augmented views
            view1 = self.strong_augment(image=image)['image']
            view2 = self.strong_augment(image=image)['image']
            return view1, view2
        
        elif self.ssl_method == 'byol':
            # Generate augmented view and target view
            augmented = self.strong_augment(image=image)['image']
            target = self.weak_augment(image=image)['image']
            return augmented, target
        
        else:  # Default to simple augmentation
            augmented = self.strong_augment(image=image)['image']
            return augmented, augmented

class SimCLRModel(nn.Module):
    """SimCLR implementation for agricultural image representation learning"""
    
    def __init__(self, base_model: str = 'resnet50', projection_dim: int = 256):
        super(SimCLRModel, self).__init__()
        
        # Load pre-trained backbone
        if base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            self.feature_dim = 2048
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)

class BYOLModel(nn.Module):
    """BYOL implementation for agricultural image representation learning"""
    
    def __init__(self, base_model: str = 'resnet50', projection_dim: int = 256):
        super(BYOLModel, self).__init__()
        
        # Online network
        self.online_encoder = models.resnet50(pretrained=True)
        self.online_encoder.fc = nn.Identity()
        
        self.online_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim)
        )
        
        # Target network (EMA of online network)
        self.target_encoder = models.resnet50(pretrained=True)
        self.target_encoder.fc = nn.Identity()
        
        self.target_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Initialize target network
        self._initialize_target_network()
        
        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _initialize_target_network(self):
        """Initialize target network with online network weights"""
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data.copy_(online_param.data)
        
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data.copy_(online_param.data)
    
    def update_target_network(self, tau: float = 0.99):
        """Update target network with EMA"""
        for online_param, target_param in zip(self.online_encoder.parameters(), 
                                            self.target_encoder.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
    
    def forward(self, x1, x2):
        # Online network
        online_features_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_features_1)
        online_pred_1 = self.online_predictor(online_proj_1)
        
        online_features_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_features_2)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # Target network
        with torch.no_grad():
            target_features_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_features_1)
            
            target_features_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_features_2)
        
        return online_pred_1, online_pred_2, target_proj_1, target_proj_2

class ContinuousLearningManager:
    """Manages continuous learning from rover's captured frames"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ssl_method = config.get('ssl_method', 'simclr')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
        # Initialize SSL model
        if self.ssl_method == 'simclr':
            self.ssl_model = SimCLRModel().to(self.device)
        elif self.ssl_method == 'byol':
            self.ssl_model = BYOLModel().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.ssl_model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize YOLO model for detection
        self.yolo_model = YOLO(config.get('yolo_model_path', 'yolo11s.pt'))
        
        # Database for storing learning progress
        self.db_path = config.get('db_path', 'ssl_learning.db')
        self._init_database()
        
        # Frame collection buffer
        self.frame_buffer = []
        self.buffer_size = config.get('buffer_size', 1000)
        
        # Learning thresholds
        self.learning_threshold = config.get('learning_threshold', 100)  # frames
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
    def _setup_logger(self):
        """Setup logging for SSL manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('SSL_Manager')
    
    def _init_database(self):
        """Initialize SQLite database for tracking learning progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                frame_path TEXT,
                ssl_loss REAL,
                confidence_score REAL,
                learned_features TEXT,
                model_version INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT,
                performance_metrics TEXT,
                version INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_frame(self, frame_path: str, detection_results: Dict):
        """Add new frame to learning buffer"""
        frame_info = {
            'path': frame_path,
            'timestamp': datetime.now(),
            'detections': detection_results,
            'confidence': detection_results.get('confidence', 0.0)
        }
        
        self.frame_buffer.append(frame_info)
        
        # Remove old frames if buffer is full
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Check if we should trigger learning
        if len(self.frame_buffer) >= self.learning_threshold:
            self._trigger_learning()
    
    def _trigger_learning(self):
        """Trigger self-supervised learning on buffered frames"""
        self.logger.info("Triggering self-supervised learning...")
        
        # Select frames for learning (focus on uncertain detections)
        learning_frames = self._select_learning_frames()
        
        if len(learning_frames) > 10:  # Minimum frames for learning
            # Perform SSL training
            ssl_loss = self._perform_ssl_training(learning_frames)
            
            # Update model if improvement is significant
            if ssl_loss < self.config.get('improvement_threshold', 0.1):
                self._update_yolo_model()
            
            # Log learning progress
            self._log_learning_progress(learning_frames, ssl_loss)
    
    def _select_learning_frames(self) -> List[str]:
        """Select frames for learning based on uncertainty and diversity"""
        # Sort by confidence (lower confidence = more uncertain)
        uncertain_frames = sorted(
            self.frame_buffer,
            key=lambda x: x['confidence']
        )
        
        # Select top uncertain frames
        selected_frames = uncertain_frames[:self.config.get('learning_batch_size', 50)]
        
        return [frame['path'] for frame in selected_frames]
    
    def _perform_ssl_training(self, frame_paths: List[str]) -> float:
        """Perform self-supervised learning on selected frames"""
        # Create dataset and dataloader
        dataset = SSLDataset(frame_paths, ssl_method=self.ssl_method)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=4
        )
        
        total_loss = 0.0
        num_batches = 0
        
        self.ssl_model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            if self.ssl_method == 'simclr':
                view1, view2 = batch
                view1, view2 = view1.to(self.device), view2.to(self.device)
                
                # Combine views
                combined = torch.cat([view1, view2], dim=0)
                features = self.ssl_model(combined)
                
                # Calculate SimCLR loss
                loss = self._simclr_loss(features, view1.size(0))
                
            elif self.ssl_method == 'byol':
                view1, view2 = batch
                view1, view2 = view1.to(self.device), view2.to(self.device)
                
                pred1, pred2, target1, target2 = self.ssl_model(view1, view2)
                
                # Calculate BYOL loss
                loss = self._byol_loss(pred1, pred2, target1, target2)
                
                # Update target network
                self.ssl_model.update_target_network()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(f"SSL Training Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"SSL Training completed. Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _simclr_loss(self, features: torch.Tensor, batch_size: int, 
                    temperature: float = 0.5) -> torch.Tensor:
        """Calculate SimCLR contrastive loss"""
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create positive pairs mask
        batch_size = features.size(0) // 2
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        mask = mask.repeat(2, 2)
        mask = mask ^ torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        
        # Calculate contrastive loss
        positive_samples = similarity_matrix[mask].view(2 * batch_size, -1)
        negative_samples = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def _byol_loss(self, pred1: torch.Tensor, pred2: torch.Tensor, 
                  target1: torch.Tensor, target2: torch.Tensor) -> torch.Tensor:
        """Calculate BYOL loss"""
        pred1 = F.normalize(pred1, dim=1)
        pred2 = F.normalize(pred2, dim=1)
        target1 = F.normalize(target1, dim=1)
        target2 = F.normalize(target2, dim=1)
        
        loss1 = 2 - 2 * (pred1 * target2).sum(dim=1)
        loss2 = 2 - 2 * (pred2 * target1).sum(dim=1)
        
        return (loss1 + loss2).mean()
    
    def _update_yolo_model(self):
        """Update YOLO model with learned features"""
        self.logger.info("Updating YOLO model with learned features...")
        
        # Extract features from SSL model
        feature_extractor = self.ssl_model.backbone if hasattr(self.ssl_model, 'backbone') else self.ssl_model.online_encoder
        
        # Create feature bank for pseudo-labeling
        feature_bank = self._create_feature_bank()
        
        # Generate pseudo-labels for unlabeled data
        pseudo_labels = self._generate_pseudo_labels(feature_bank)
        
        # Fine-tune YOLO model with pseudo-labels
        if len(pseudo_labels) > 0:
            self._fine_tune_yolo(pseudo_labels)
    
    def _create_feature_bank(self) -> Dict:
        """Create feature bank from SSL model"""
        feature_bank = {}
        
        # Extract features from recent frames
        recent_frames = self.frame_buffer[-100:]  # Last 100 frames
        
        self.ssl_model.eval()
        with torch.no_grad():
            for frame_info in recent_frames:
                frame_path = frame_info['path']
                
                # Load and preprocess image
                image = cv2.imread(frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing
                transform = A.Compose([
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                
                image_tensor = transform(image=image)['image'].unsqueeze(0).to(self.device)
                
                # Extract features
                if hasattr(self.ssl_model, 'backbone'):
                    features = self.ssl_model.backbone(image_tensor)
                else:
                    features = self.ssl_model.online_encoder(image_tensor)
                
                feature_bank[frame_path] = features.cpu().numpy()
        
        return feature_bank
    
    def _generate_pseudo_labels(self, feature_bank: Dict) -> List[Dict]:
        """Generate pseudo-labels using clustering"""
        if not feature_bank:
            return []
        
        # Prepare features for clustering
        features = np.array(list(feature_bank.values())).squeeze()
        frame_paths = list(feature_bank.keys())
        
        # Perform clustering
        n_clusters = min(10, len(features))  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Generate pseudo-labels
        pseudo_labels = []
        for i, frame_path in enumerate(frame_paths):
            pseudo_labels.append({
                'image_path': frame_path,
                'cluster_id': cluster_labels[i],
                'confidence': float(np.max(kmeans.transform([features[i]])))
            })
        
        return pseudo_labels
    
    def _fine_tune_yolo(self, pseudo_labels: List[Dict]):
        """Fine-tune YOLO model with pseudo-labels"""
        # This would involve creating a new training dataset with pseudo-labels
        # and fine-tuning the YOLO model
        self.logger.info(f"Fine-tuning YOLO with {len(pseudo_labels)} pseudo-labels")
        
        # Save current model as checkpoint
        checkpoint_path = f"models/yolo_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        # self.yolo_model.save(checkpoint_path)
        
        # Log checkpoint
        self._log_model_checkpoint(checkpoint_path)
    
    def _log_learning_progress(self, frame_paths: List[str], ssl_loss: float):
        """Log learning progress to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for frame_path in frame_paths:
            cursor.execute('''
                INSERT INTO learning_progress 
                (frame_path, ssl_loss, confidence_score, learned_features, model_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (frame_path, ssl_loss, 0.0, "", 1))
        
        conn.commit()
        conn.close()
    
    def _log_model_checkpoint(self, checkpoint_path: str):
        """Log model checkpoint to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_checkpoints 
            (model_path, performance_metrics, version)
            VALUES (?, ?, ?)
        ''', (checkpoint_path, "{}", 1))
        
        conn.commit()
        conn.close()
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM learning_progress')
        total_frames = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(ssl_loss) FROM learning_progress')
        avg_loss = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT COUNT(*) FROM model_checkpoints')
        total_checkpoints = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_frames_learned': total_frames,
            'average_ssl_loss': avg_loss,
            'total_checkpoints': total_checkpoints,
            'current_buffer_size': len(self.frame_buffer)
        }

class TerraRoverSSLIntegration:
    """Integration class for Terra Rover SSL system"""
    
    def __init__(self, config_path: str = "config/ssl_config.json"):
        self.config = self._load_config(config_path)
        self.ssl_manager = ContinuousLearningManager(self.config)
        self.yolo_model = YOLO(self.config.get('yolo_model_path', 'yolo11s.pt'))
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'ssl_method': 'simclr',
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'batch_size': 16,
                'buffer_size': 1000,
                'learning_threshold': 100,
                'confidence_threshold': 0.5,
                'improvement_threshold': 0.1,
                'learning_batch_size': 50,
                'yolo_model_path': 'yolo11s.pt',
                'db_path': 'ssl_learning.db'
            }
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('TerraRoverSSL')
    
    def process_frame(self, frame_path: str) -> Dict:
        """Process a single frame with YOLO detection and SSL integration"""
        # Run YOLO detection
        results = self.yolo_model(frame_path)
        
        # Extract detection information
        detection_info = self._extract_detection_info(results)
        
        # Add frame to SSL learning buffer
        self.ssl_manager.add_frame(frame_path, detection_info)
        
        return detection_info
    
    def _extract_detection_info(self, results) -> Dict:
        """Extract detection information from YOLO results"""
        if not results:
            return {'confidence': 0.0, 'detections': []}
        
        detections = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf)
                    confidences.append(confidence)
                    
                    detection = {
                        'class': int(box.cls),
                        'confidence': confidence,
                        'bbox': box.xyxy.tolist()[0] if box.xyxy is not None else []
                    }
                    detections.append(detection)
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'confidence': avg_confidence,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def get_system_status(self) -> Dict:
        """Get system status and learning statistics"""
        stats = self.ssl_manager.get_learning_statistics()
        
        return {
            'ssl_stats': stats,
            'yolo_model_loaded': self.yolo_model is not None,
            'config': self.config
        }

# Example usage and integration
def main():
    """Example usage of the SSL Terra Rover system"""
    
    # Initialize SSL system
    ssl_system = TerraRoverSSLIntegration()
    
    sample_frame_path = "data/captured_frames/sample_frame.jpg"
    
    if os.path.exists(sample_frame_path):
        # Process frame
        detection_results = ssl_system.process_frame(sample_frame_path)
        print(f"Detection results: {detection_results}")
        
        # Get system status
        status = ssl_system.get_system_status()
        print(f"System status: {status}")

if __name__ == "__main__":
    main()
