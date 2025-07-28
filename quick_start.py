"""
Quick start script for AI-Generated Image Detection
"""

import os
import sys
import argparse
from pathlib import Path

def create_sample_structure():
    """Create sample directory structure"""
    print("Creating sample directory structure...")
    
    # Create directories
    directories = [
        "data/real",
        "data/fake", 
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")
    
    # Create sample README for data directory
    data_readme = """# Dataset Directory

Place your images in the following structure:

## Real Images
Put real (non-AI-generated) images in the `real/` folder.

## AI-Generated Images  
Put AI-generated images in the `fake/` folder.

## Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)

## Image Size
Images will be automatically resized to 32x32 pixels during training.
"""
    
    with open("data/README.md", "w") as f:
        f.write(data_readme)
    
    print("✓ Created data/README.md with instructions")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    # Map pip package names to import names
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("Pillow", "PIL"),
        ("tensorboard", "tensorboard"),
        ("albumentations", "albumentations"),
        ("opencv-python", "cv2"),
        ("pandas", "pandas"),
    ]
    
    missing_packages = []
    
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {pip_name}")
        except ImportError:
            print(f"✗ {pip_name} (missing)")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Train the model:")
    print("   python train.py --data_dir data")
    
    print("\n2. Train with GPU (if available):")
    print("   python train.py --data_dir data --device cuda")
    
    print("\n3. Train with CPU:")
    print("   python train.py --data_dir data --device cpu")
    
    print("\n4. Make predictions:")
    print("   python inference.py --image_path path/to/image.jpg")
    
    print("\n5. Monitor training with TensorBoard:")
    print("   tensorboard --logdir logs")
    
    print("\n6. Resume training from checkpoint:")
    print("   python train.py --data_dir data --resume models/checkpoint_epoch_50.pth")

def main():
    parser = argparse.ArgumentParser(description='Quick start for AI-Generated Image Detection')
    parser.add_argument('--setup', action='store_true', help='Create directory structure')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')
    parser.add_argument('--examples', action='store_true', help='Show usage examples')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    if not any([args.setup, args.check_deps, args.examples, args.all]):
        print("AI-Generated Image Detection - Quick Start")
        print("="*50)
        print("\nUse --help to see available options:")
        print("python quick_start.py --help")
        print("\nOr run all setup steps:")
        print("python quick_start.py --all")
        return
    
    if args.all or args.setup:
        create_sample_structure()
    
    if args.all or args.check_deps:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("\nPlease install missing dependencies before proceeding.")
            return
    
    if args.all or args.examples:
        show_usage_examples()
    
    if args.all:
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Add your images to data/real/ and data/fake/ folders")
        print("2. Run: python train.py --data_dir data")
        print("3. Monitor training with: tensorboard --logdir logs")
        print("4. Make predictions with: python inference.py --image_path your_image.jpg")

if __name__ == '__main__':
    main() 