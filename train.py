"""
Main training script for Vision Transformer on CIFake dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

from config import Config
from vision_transformer import create_vit_model
from dataset import create_dataloaders, cleanup_temp_dirs
from utils import (
    LabelSmoothingLoss, MixupLoss, mixup_data, cutmix_data,
    WarmupCosineScheduler, calculate_metrics, plot_confusion_matrix,
    plot_training_history, save_checkpoint, load_checkpoint, evaluate_model,
    count_parameters, set_seed
)


def train_epoch(model, train_loader, optimizer, criterion, device, config, epoch):
    """Train for one epoch with debug output and error handling"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        try:
            data, target = batch
            print(f"Processing batch {batch_idx} (data shape: {data.shape}, target shape: {target.shape})")
            data, target = data.to(device), target.to(device)
            # Apply data augmentation (Mixup/CutMix)
            if np.random.random() < 0.5:
                if np.random.random() < 0.5:
                    # Mixup
                    data, target_a, target_b, lam = mixup_data(data, target, config.MIXUP_ALPHA)
                    criterion_mixup = MixupLoss(criterion)
                    output = model(data)
                    loss = criterion_mixup(output, target_a, target_b, lam)
                else:
                    # CutMix
                    data, target_a, target_b, lam = cutmix_data(data, target, config.CUTMIX_ALPHA)
                    criterion_mixup = MixupLoss(criterion)
                    output = model(data)
                    loss = criterion_mixup(output, target_a, target_b, lam)
            else:
                # No augmentation
                output = model(data)
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            if 'lam' in locals():
                correct += lam * pred.eq(target_a).sum().item() + (1 - lam) * pred.eq(target_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            total += target.size(0)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    return total_loss / len(train_loader), 100. * correct / total


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFake dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override device if specified
    if args.device:
        config.DEVICE = args.device
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Create model
    model = create_vit_model(config)
    model = model.to(device)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, args.data_dir)
    
    # Loss function and optimizer
    criterion = LabelSmoothingLoss(config.NUM_CLASSES, config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        config.WARMUP_EPOCHS, 
        config.NUM_EPOCHS
    )
    
    # TensorBoard writer
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        start_epoch, checkpoint_metrics = load_checkpoint(model, optimizer, args.resume)
        best_val_acc = checkpoint_metrics.get('best_val_acc', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(epoch)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, 
                {'best_val_acc': best_val_acc}, 
                os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
            )
            print(f'  New best validation accuracy: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, 
                {'best_val_acc': best_val_acc}, 
                os.path.join(config.MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'Early stopping after {config.PATIENCE} epochs without improvement')
            break
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics, test_preds, test_targets, test_probs = evaluate_model(
        model, test_loader, device, criterion
    )
    
    print("\nFinal Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot results
    print("Generating plots...")
    plot_training_history(history, os.path.join(config.LOG_DIR, 'training_history.png'))
    plot_confusion_matrix(test_targets, test_preds, os.path.join(config.LOG_DIR, 'confusion_matrix.png'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'final_model.pth'))
    
    # Cleanup
    cleanup_temp_dirs(args.data_dir)
    writer.close()
    
    print("Training completed!")


if __name__ == '__main__':
    main() 