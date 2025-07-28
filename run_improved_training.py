#!/usr/bin/env python3
"""
Script to run improved training for better accuracy
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("✗ No GPU available, will use CPU")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_data():
    """Check if data directory exists and has images"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ Data directory not found")
        return False
    
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        print("✗ Data directory structure incorrect. Need 'data/real' and 'data/fake' folders")
        return False
    
    real_count = len(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    fake_count = len(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    
    if real_count == 0 or fake_count == 0:
        print("✗ No images found in data directories")
        return False
    
    print(f"✓ Data found: {real_count} real images, {fake_count} fake images")
    return True

def run_training(device=None, resume=None, experiment_name="improved_vit"):
    """Run the improved training"""
    
    cmd = [
        sys.executable, "train_improved.py",
        "--data_dir", "data",
        "--experiment_name", experiment_name
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    if resume:
        cmd.extend(["--resume", resume])
    
    print(f"Running command: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("STARTING IMPROVED TRAINING")
    print("="*60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run improved training for better accuracy')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default='improved_vit',
                       help='Experiment name for logging')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip GPU and data checks')
    
    args = parser.parse_args()
    
    print("AI-Generated Image Detection - Improved Training")
    print("="*50)
    
    # Perform checks
    if not args.skip_checks:
        print("\nPerforming pre-training checks...")
        
        # Check GPU
        gpu_available = check_gpu()
        
        # Check data
        data_ok = check_data()
        
        if not data_ok:
            print("\nPlease ensure your data is properly set up before running training.")
            print("Expected structure:")
            print("  data/")
            print("  ├── real/     (real images)")
            print("  └── fake/     (AI-generated images)")
            return
        
        print("\n✓ All checks passed!")
    
    # Determine device
    if not args.device:
        if check_gpu():
            args.device = "cuda"
        else:
            args.device = "cpu"
    
    print(f"\nWill use device: {args.device}")
    
    # Show configuration summary
    print("\nConfiguration Summary:")
    print("- Image size: 224x224 (increased from 32x32)")
    print("- Model capacity: 768 embed dim, 12 heads (increased)")
    print("- Training epochs: 50 (increased from 5)")
    print("- Batch size: 32 (increased from 16)")
    print("- Enhanced data augmentation")
    print("- Gradient clipping and improved learning rate scheduling")
    
    # Ask for confirmation
    response = input("\nProceed with training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Run training
    success = run_training(
        device=args.device,
        resume=args.resume,
        experiment_name=args.experiment_name
    )
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the logs directory for training plots")
        print("2. Use the improved model for inference:")
        print("   python inference.py --model_path models/best_model_improved.pth")
        print("3. Monitor training with TensorBoard:")
        print(f"   tensorboard --logdir logs/{args.experiment_name}")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == '__main__':
    main() 