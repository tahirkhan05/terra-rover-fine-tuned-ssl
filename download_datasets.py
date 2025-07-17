import os
import sys
import shutil
import zipfile
from pathlib import Path
import subprocess
import json

class KaggleDatasetDownloader:
    """Download and organize Kaggle datasets for SSL testing"""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations matching test_ssl_datasets.py
        self.datasets = {
            'new_plant_diseases': {
                'kaggle_id': 'vipoooool/new-plant-diseases-dataset',
                'local_path': 'datasets/new_plant_diseases/',
                'description': 'Plant disease classification dataset with 38 classes'
            },
            'agriculture_crops': {
                'kaggle_id': 'aman2000jaiswal/agriculture-crop-images',
                'local_path': 'datasets/agriculture_crops/',
                'description': 'Agricultural crop type classification'
            },
            'crop_weed_detection': {
                'kaggle_id': 'ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
                'local_path': 'datasets/crop_weed_detection/',
                'description': 'Crop and weed detection with bounding boxes'
            },
            'agricultural_crops_classification': {
                'kaggle_id': 'mdwaquarazam/agricultural-crops-image-classification',
                'local_path': 'datasets/agricultural_crops_classification/',
                'description': 'Agricultural crops classification dataset'
            }
        }
    
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly set up"""
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Kaggle CLI found: {result.stdout.strip()}")
            
            # Test API authentication
            result = subprocess.run(['kaggle', 'datasets', 'list', '--max-size', '1'], 
                                  capture_output=True, text=True, check=True)
            print("✅ Kaggle API authentication successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Kaggle setup error: {e}")
            print("\n🔧 Setup Instructions:")
            print("1. Install kaggle: pip install kaggle")
            print("2. Get API token from: https://www.kaggle.com/account")
            print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        except FileNotFoundError:
            print("❌ Kaggle CLI not found")
            print("Install with: pip install kaggle")
            return False
    
    def download_dataset(self, dataset_id: str, dataset_config: dict) -> bool:
        """Download a specific dataset from Kaggle"""
        print(f"\n📥 Downloading {dataset_id}...")
        print(f"   Description: {dataset_config['description']}")
        
        local_path = Path(dataset_config['local_path'])
        
        # Check if already downloaded
        if local_path.exists() and any(local_path.iterdir()):
            print(f"   ✅ Already exists at {local_path}")
            return True
        
        # Create directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download dataset
            print(f"   🔄 Downloading {dataset_config['kaggle_id']}...")
            result = subprocess.run([
                'kaggle', 'datasets', 'download', 
                dataset_config['kaggle_id'],
                '--path', str(local_path),
                '--unzip'
            ], capture_output=True, text=True, check=True)
            
            print(f"   ✅ Downloaded and extracted to {local_path}")
            
            # Verify download
            if any(local_path.iterdir()):
                file_count = sum(1 for _ in local_path.rglob('*') if _.is_file())
                print(f"   📊 Found {file_count} files")
                return True
            else:
                print(f"   ⚠️ Download completed but no files found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Download failed: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    def organize_dataset(self, dataset_id: str) -> bool:
        """Organize dataset structure for consistent access"""
        print(f"\n🗂️ Organizing {dataset_id}...")
        
        local_path = Path(self.datasets[dataset_id]['local_path'])
        
        if not local_path.exists():
            print(f"   ❌ Dataset not found at {local_path}")
            return False
        
        try:
            # Count images before organization
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            before_count = sum(1 for f in local_path.rglob('*') 
                             if f.is_file() and f.suffix.lower() in image_extensions)
            
            print(f"   📸 Found {before_count} images")
            
            # Create organized structure summary
            subdirs = [d for d in local_path.iterdir() if d.is_dir()]
            print(f"   📁 Directory structure: {len(subdirs)} subdirectories")
            
            # Create a dataset info file
            info_file = local_path / "dataset_info.json"
            info = {
                'dataset_id': dataset_id,
                'image_count': before_count,
                'subdirectories': [d.name for d in subdirs],
                'organized_at': str(Path().absolute()),
                'structure_type': 'class_folders' if len(subdirs) > 1 else 'flat'
            }
            
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"   ✅ Organization complete")
            return True
            
        except Exception as e:
            print(f"   ❌ Organization failed: {e}")
            return False
    
    def create_dataset_summary(self):
        """Create a summary of all downloaded datasets"""
        print(f"\n📋 Creating dataset summary...")
        
        summary = {
            'download_summary': {},
            'total_datasets': 0,
            'total_images': 0,
            'datasets_ready': []
        }
        
        for dataset_id, config in self.datasets.items():
            local_path = Path(config['local_path'])
            
            if local_path.exists():
                # Count images
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_count = sum(1 for f in local_path.rglob('*') 
                                if f.is_file() and f.suffix.lower() in image_extensions)
                
                summary['download_summary'][dataset_id] = {
                    'status': 'downloaded',
                    'path': str(local_path),
                    'image_count': image_count,
                    'description': config['description']
                }
                
                if image_count > 0:
                    summary['datasets_ready'].append(dataset_id)
                    summary['total_images'] += image_count
                    summary['total_datasets'] += 1
            else:
                summary['download_summary'][dataset_id] = {
                    'status': 'not_downloaded',
                    'path': str(local_path),
                    'description': config['description']
                }
        
        # Save summary
        summary_file = self.base_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   💾 Summary saved to: {summary_file}")
        print(f"   📊 Total datasets ready: {summary['total_datasets']}")
        print(f"   📸 Total images: {summary['total_images']}")
        
        return summary
    
    def download_all_datasets(self):
        """Download and organize all datasets"""
        print("🚀 Starting Kaggle Dataset Download for Terra Rover SSL Testing")
        print("=" * 80)
        
        # Check Kaggle setup
        if not self.check_kaggle_setup():
            return False
        
        print(f"\n📁 Base directory: {self.base_dir.absolute()}")
        
        # Download each dataset
        success_count = 0
        for dataset_id, config in self.datasets.items():
            if self.download_dataset(dataset_id, config):
                if self.organize_dataset(dataset_id):
                    success_count += 1
        
        # Create summary
        summary = self.create_dataset_summary()
        
        print(f"\n🎉 Download Complete!")
        print("=" * 80)
        print(f"✅ Successfully downloaded: {success_count}/{len(self.datasets)} datasets")
        print(f"📸 Total images ready for SSL testing: {summary['total_images']}")
        
        if success_count > 0:
            print(f"\n🧪 Ready to run SSL tests with:")
            for dataset_id in summary['datasets_ready']:
                print(f"   - {dataset_id}: {summary['download_summary'][dataset_id]['image_count']} images")
            
            print(f"\n🏃 Run SSL tests with: python test_ssl_datasets.py")
        
        return success_count > 0

def main():
    """Main download function"""
    downloader = KaggleDatasetDownloader()
    
    # Check if user wants to download all or specific datasets
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            # Just check what's already downloaded
            summary = downloader.create_dataset_summary()
            return
        elif sys.argv[1] == "--setup":
            # Show setup instructions
            print("🔧 Kaggle API Setup Instructions")
            print("=" * 50)
            print("1. Install Kaggle package:")
            print("   pip install kaggle")
            print("\n2. Get your API token:")
            print("   - Go to https://www.kaggle.com/account")
            print("   - Click 'Create New Token'")
            print("   - Download kaggle.json")
            print("\n3. Place token file:")
            print("   Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
            print("   Linux/Mac: ~/.kaggle/kaggle.json")
            print("\n4. Set permissions (Linux/Mac only):")
            print("   chmod 600 ~/.kaggle/kaggle.json")
            print("\n5. Test setup:")
            print("   kaggle datasets list --max-size 1")
            return
    
    # Download all datasets
    downloader.download_all_datasets()

if __name__ == "__main__":
    main()
