"""
Improved training script for Vision Transformer on CIFake dataset
Enhanced with better learning rate scheduling, gradient clipping, and advanced techniques
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


class ImprovedWarmupCosineScheduler:
    """Improved learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing with minimum learning rate
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch_improved(model, train_loader, optimizer, criterion, device, config, epoch, scheduler):
    """Improved training for one epoch with gradient clipping and better monitoring"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        try:
            data, target = batch
            data, target = data.to(device), target.to(device)
            
            # Apply data augmentation (Mixup/CutMix) with adaptive probability
            if np.random.random() < 0.7:  # Increased probability
                if np.random.random() < 0.6:
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            if 'lam' in locals():
                correct += lam * pred.eq(target_a).sum().item() + (1 - lam) * pred.eq(target_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / len(train_loader), 100. * correct / total


def validate_epoch_improved(model, val_loader, criterion, device):
    """Improved validation with better metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    return total_loss / len(val_loader), accuracy, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFake dataset (Improved)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--experiment_name', type=str, default='improved_vit', help='Experiment name for logging')
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
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Improved learning rate scheduler
    scheduler = ImprovedWarmupCosineScheduler(
        optimizer, 
        config.WARMUP_EPOCHS, 
        config.NUM_EPOCHS
    )
    
    # TensorBoard writer
    log_dir = os.path.join(config.LOG_DIR, args.experiment_name)
    writer = SummaryWriter(log_dir)
    
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
    print("Starting improved training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch_improved(
            model, train_loader, optimizer, criterion, device, config, epoch, scheduler
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch_improved(
            model, val_loader, criterion, device
        )
        
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
        
        # Log validation predictions distribution
        writer.add_histogram('Validation/Predictions', torch.tensor(val_preds), epoch)
        writer.add_histogram('Validation/Targets', torch.tensor(val_targets), epoch)
        
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
                os.path.join(config.MODEL_SAVE_DIR, 'best_model_improved.pth')
            )
            print(f'  New best validation accuracy: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, 
                {'best_val_acc': best_val_acc}, 
                os.path.join(config.MODEL_SAVE_DIR, f'checkpoint_improved_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'Early stopping after {config.PATIENCE} epochs without improvement')
            break
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(config.MODEL_SAVE_DIR, 'best_model_improved.pth'))
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
    plot_training_history(history, os.path.join(config.LOG_DIR, 'training_history_improved.png'))
    plot_confusion_matrix(test_targets, test_preds, os.path.join(config.LOG_DIR, 'confusion_matrix_improved.png'))
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS - 1,
        {'best_val_acc': best_val_acc, 'test_metrics': test_metrics},
        os.path.join(config.MODEL_SAVE_DIR, 'final_model_improved.pth')
    )
    
    # Cleanup
    cleanup_temp_dirs(args.data_dir)
    writer.close()
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODEL_SAVE_DIR}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main() 