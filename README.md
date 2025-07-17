# Terra Rover Plant Disease Detection System with SSL

A real-time plant disease detection system that uses fine-tuned YOLO models combined with Self-Supervised Learning (SSL) for automated monitoring and continuous improvement in agricultural field analysis.

## Overview

Terra Rover is an integrated system that combines computer vision, voice recognition, vision-language models, and Self-Supervised Learning (SSL) to detect and analyze plant diseases in real-time. The system processes RTSP video streams, identifies plant diseases and pests, and continuously improves its detection accuracy through automated learning from captured frames. Users can also ask questions about detected plants through voice queries, making it a comprehensive agricultural monitoring solution.

## Features

- **Real-time plant disease detection** using fine-tuned YOLO11s models
- **Self-Supervised Learning (SSL)** for continuous model improvement without manual labeling
- **Voice and Text query interface** for asking questions about plants in view
- **AWS integration** for cloud-based image storage and analysis
- **Vision-Language Model support** for detailed plant analysis
- **Optimized video processing pipeline** for high-performance operation
- **Adaptive learning** from uncertain detections to improve accuracy over time
- **Background processing** for SSL training without interrupting main operations

## Architecture

```
terra-rover-fine-tuned/
├── config/             # Configuration settings
├── data/               # Local storage for captured frames
├── models/             # ML model implementations
├── services/           # Core service components
├── utils/              # Utility functions
├── main.py             # Main application entry point
├── model_loader.py     # Model loading utilities
├── requirements.txt    # Project dependencies
├── ssl_learning.db     # SSL learning progress database
└── README.md           # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/tahirkhan05/terra-rover-fine-tuned.git
   cd terra-rover-fine-tuned
   ```

2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables by editing the `.env` file:
   ```
   # AWS Credentials
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   S3_BUCKET=terra-rover-bucket 

   # RTSP Stream (use IP Webcam app for testing)
   RTSP_URL=rtsp://your_camera_ip/h264_pcm.sdp
   RTSP_RECONNECT_DELAY=2
   RTSP_MAX_RETRIES=10
   RTSP_MAX_CONSECUTIVE_FAILURES=30

   # High FPS Configuration
   FPS=60
   FRAME_WIDTH=1280
   FRAME_HEIGHT=720

   # Model Settings
   OBJECT_DETECTION_MODEL=yolo11s.pt
   VLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

   # Speech Configuration
   LEX_BOT_ID=your_lex_bot_id
   LEX_BOT_ALIAS_ID=your_lex_alias_id
   LEX_LOCALE_ID=en_US

   # System Configuration
   MAX_WORKERS=8
   LOG_LEVEL=INFO  #use DEBUG for detailed logs
   DEBUG=True
   LOCAL_SAVE_PATH=data/captured_frames

   # Performance Tuning
   MAX_QUEUE_SIZE=60
   PROCESSING_INTERVAL=0.01
   DETECTION_CONF_THRESHOLD=0.25
   IOU_THRESHOLD=0.45
   ```

5. Prepare the fine-tuned YOLO model:
   - The system expects a fine-tuned YOLO model in the root directory
   - This model was fine-tuned using [https://github.com/tahirkhan05/yolo-fine-tuner](https://github.com/tahirkhan05/yolo-fine-tuner)
   - Place your fine-tuned model file (e.g., `best.pt`) in the project root

## Usage

1. Start the application:
   ```
   python main.py
   ```

2. Using the interface:
   - Press 's' to activate voice query mode
   - Or press 't' to input prompt into text field
   - Press 'l' to toggle SSL background learning
   - Press 'p' to toggle SSL frame processing
   - Press 'r' to view SSL system status
   - Ask questions about plants in view when prompted
   - Press 'q' to quit the application

## Voice Query Examples

- "What plant diseases are visible?"
- "Are there any weeds in this field?"
- "Can you identify the plants in this image?"
- "How serious is the leaf disease in the top-right?"

## Self-Supervised Learning (SSL) Features

Terra Rover includes advanced SSL capabilities that enable the system to continuously learn and improve from captured frames without requiring manual labeling.

### Key SSL Features

- **Continuous Learning**: Automatically learns from frames captured during operation
- **Uncertainty-Based Selection**: Prioritizes learning from frames with uncertain detections (below confidence threshold)
- **Background Processing**: SSL training runs in the background without interrupting main detection operations
- **Multiple SSL Methods**: Supports SimCLR (contrastive learning) and BYOL (bootstrap learning) frameworks
- **Adaptive Thresholds**: Dynamically adjusts learning parameters based on system performance
- **Model Checkpointing**: Automatically saves improved models for future use

### SSL Status Monitoring

Press 'r' to view real-time SSL status including:
- SSL learning active/inactive status
- Current buffer size and total frames processed
- SSL loss values and model checkpoint information
- Learning session statistics and average confidence scores

## Dependencies

- OpenCV (computer vision and video processing)
- PyTorch (deep learning framework for SSL and YOLO)
- Ultralytics YOLO (object detection models)
- AWS SDK (boto3) (cloud integration)
- SoundDevice (audio capture for voice queries)
- NumPy and SciPy (numerical computing)
- SQLite (SSL learning progress tracking)
- scikit-learn (machine learning utilities for SSL)
- Other dependencies listed in requirements.txt

## AWS Services Used

- S3 (image storage)
- Bedrock (Vision-Language Model inference)
- Amazon Lex (speech recognition)

## Model Fine-Tuning

The plant disease detection model was fine-tuned using [YOLO Fine-Tuner](https://github.com/tahirkhan05/yolo-fine-tuner) with a custom dataset of plant diseases, pests, and weeds. The fine-tuning process produces a specialized model that can detect:

- Various plant diseases (leaf spots, blights, etc.)
- Common agricultural weeds
- Plant stress indicators

The fine-tuned model should be placed in the root directory of the project and referenced in the `.env` configuration file.

## Performance Optimization

The system is optimized for real-time processing with several performance enhancements:
- Dynamic frame skipping based on system load
- Hardware acceleration when available
- Parallel processing for non-critical tasks
- Latency monitoring and adaptation

## SSL Learning and Database

### Learning Progress Tracking

The SSL system uses `ssl_learning.db` SQLite database to track:
- Frame processing history and timestamps
- SSL loss values and improvement metrics
- Model checkpoint information and versions
- Learning session statistics and performance data

### Background Learning Process

- **Non-blocking Design**: SSL learning runs in separate threads to avoid interrupting main detection
- **Intelligent Buffering**: Efficiently manages frame buffers for optimal memory usage
- **Adaptive Learning**: Adjusts learning frequency based on detection uncertainty and system performance
- **Checkpoint Management**: Automatically saves model improvements and maintains version history

### SSL Performance Tips

- **GPU Acceleration**: SSL training benefits significantly from GPU acceleration
- **Memory Requirements**: At least 8GB RAM recommended for smooth SSL operation
- **Storage**: Ensure sufficient disk space for frame storage and model checkpoints
- **Configuration Tuning**: Start with default settings and adjust based on your specific use case