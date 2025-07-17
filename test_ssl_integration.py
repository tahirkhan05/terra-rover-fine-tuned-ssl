"""
SSL Integration Test Script for Terra Rover
Tests the SSL framework components and integration
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def test_ssl_config():
    """Test SSL configuration loading"""
    print("Testing SSL Configuration...")
    
    config_path = "config/ssl_config.json"
    if not os.path.exists(config_path):
        print(f"❌ SSL config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['ssl_method', 'learning_rate', 'batch_size', 'yolo_model_path']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        print(f"✅ SSL config loaded successfully")
        print(f"   SSL Method: {config['ssl_method']}")
        print(f"   Learning Rate: {config['learning_rate']}")
        print(f"   Batch Size: {config['batch_size']}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading SSL config: {str(e)}")
        return False

def test_ssl_framework():
    """Test SSL framework import and initialization"""
    print("\nTesting SSL Framework...")
    
    try:
        from ssl_framework import TerraRoverSSLIntegration, ContinuousLearningManager
        print("✅ SSL framework imported successfully")
        
        # Test initialization
        ssl_system = TerraRoverSSLIntegration()
        print("✅ SSL system initialized successfully")
        
        # Test status
        status = ssl_system.get_system_status()
        print(f"✅ SSL status retrieved: {len(status)} status fields")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SSL framework: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error initializing SSL system: {str(e)}")
        return False

def test_ssl_service():
    """Test SSL service integration"""
    print("\nTesting SSL Service...")
    
    try:
        from services.ssl_service import SSLService
        print("✅ SSL service imported successfully")
        
        # Test initialization
        ssl_service = SSLService()
        print("✅ SSL service initialized successfully")
        
        # Test status
        status = ssl_service.get_ssl_status()
        ssl_available = status.get('ssl_available', False)
        print(f"✅ SSL service status: Available={ssl_available}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SSL service: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error with SSL service: {str(e)}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\nTesting Dependencies...")
    
    dependencies = [
        'torch',
        'torchvision', 
        'albumentations',
        'sklearn',
        'cv2',
        'numpy',
        'ultralytics'
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - not installed")
            all_ok = False
    
    return all_ok

def test_frame_processing():
    """Test frame processing with SSL"""
    print("\nTesting Frame Processing...")
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Save test frame
    os.makedirs("data/captured_frames", exist_ok=True)
    test_path = "data/captured_frames/test_frame.jpg"
    cv2.imwrite(test_path, test_frame)
    
    if os.path.exists(test_path):
        print(f"✅ Test frame created: {test_path}")
        
        try:
            from services.ssl_service import SSLService
            ssl_service = SSLService()
            
            # Test frame processing
            result = ssl_service.process_frame_with_ssl(test_path)
            
            if 'ssl_enabled' in result:
                print(f"✅ Frame processed with SSL: {result['ssl_enabled']}")
            else:
                print("⚠️  Frame processed but SSL status unclear")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
            return False
    else:
        print("❌ Failed to create test frame")
        return False

def test_model_files():
    """Test availability of model files"""
    print("\nTesting Model Files...")
    
    model_files = ['best.pt', 'yolo11s.pt']
    found_model = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model: {model_file}")
            found_model = True
        else:
            print(f"⚠️  Model not found: {model_file}")
    
    if not found_model:
        print("❌ No YOLO models found - SSL may not work properly")
        return False
    
    return True

def test_directories():
    """Test required directories"""
    print("\nTesting Directories...")
    
    required_dirs = [
        'config',
        'models',
        'services', 
        'data/captured_frames',
        'models/ssl_checkpoints'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_ok = False
    
    return all_ok

def main():
    """Run all SSL integration tests"""
    print("="*60)
    print("🧪 Terra Rover SSL Integration Tests")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Directories", test_directories),
        ("Model Files", test_model_files),
        ("SSL Config", test_ssl_config),
        ("SSL Framework", test_ssl_framework),
        ("SSL Service", test_ssl_service),
        ("Frame Processing", test_frame_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! SSL integration is ready to use.")
        print("\nTo run Terra Rover with SSL:")
        print("  python main.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Ensure YOLO model files are present")
        print("  - Check directory permissions")
    
    print("="*60)

if __name__ == "__main__":
    main()
