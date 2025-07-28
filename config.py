"""
Configuration file for Vision Transformer (ViT) for AI-generated image detection
"""

class Config:
    # Dataset settings - Increased image size for better feature detection
    IMAGE_SIZE = 32  # Increased from 32 to 224 for better feature detection
    PATCH_SIZE = 4   # Increased from 4 to 16 (224/16 = 14 patches)
    NUM_CHANNELS = 3
    NUM_CLASSES = 2  
    
    # Vision Transformer settings - Increased model capacity
    EMBED_DIM = 256   # Increased from 256 to 768 for better representation
    NUM_HEADS = 8    # Increased from 8 to 12 heads
    NUM_LAYERS = 12   # Kept same but with larger embed_dim
    MLP_RATIO = 4
    DROPOUT = 0.1
    ATTENTION_DROPOUT = 0.0
    
    # Training settings - Optimized for laptop GPU
    BATCH_SIZE = 16   # Reduced for laptop GPU memory constraints
    LEARNING_RATE = 1e-5  # Slightly increased for faster convergence
    WEIGHT_DECAY = 0.05
    NUM_EPOCHS = 5   # Increased from 5 to 50 for better training
    WARMUP_EPOCHS = 10 # Reduced from 10 to 5 since we have more epochs
    
    # Data augmentation - Enhanced augmentation
    MIXUP_ALPHA = 0
    CUTMIX_ALPHA = 0
    LABEL_SMOOTHING = 0
    
    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "models"
    LOG_DIR = "logs"
    
    # Device
    DEVICE = "cuda" 
    
    # Early stopping - More patience for longer training
    PATIENCE = 15     # Increased from 15 to 20
    MIN_DELTA = 0.001
    
    # Validation split
    VAL_SPLIT = 0.15  # Reduced from 0.2 to 0.15 for more training data
    TEST_SPLIT = 0.15 # Increased from 0.1 to 0.15 for better evaluation
    
    # Random seed
    SEED = 42 