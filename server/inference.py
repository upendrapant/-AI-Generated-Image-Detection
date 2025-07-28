import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from vision_transformer import create_vit_model

def load_model(model_path, config):
    model = create_vit_model(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image_from_bytes(image_bytes, config):
    image = Image.open(image_bytes).convert('RGB')
    transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0)
    return image_tensor

def predict_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item()
        ai_generated_prob = probabilities[0][1].item()
    return prediction, confidence, ai_generated_prob

def analyze_image_bytes(model, image_bytes, config, device):
    image_tensor = preprocess_image_from_bytes(image_bytes, config)
    pred, conf, ai_prob = predict_image(model, image_tensor, device)
    return {
        "prediction": "AI-Generated" if pred == 1 else "Real",
        "confidence": conf,
        "ai_generated_probability": ai_prob,
        "real_probability": 1 - ai_prob
    } 