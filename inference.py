import csv
import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from vision_transformer import create_vit_model

def load_model(model_path, config):
    """Load trained model"""
    model = create_vit_model(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(image_path, config):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
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

def batch_inference(model, image_dir, config, device, output_csv="batch_predictions.csv"):
    """Run inference on all images in a directory and save results to CSV"""
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    results = []
    for img_path in image_paths:
        try:
            image_tensor = preprocess_image(img_path, config)
            pred, conf, ai_prob = predict_image(model, image_tensor, device)
            results.append({
                "image": os.path.basename(img_path),
                "prediction": "AI-Generated" if pred == 1 else "Real",
                "confidence": conf,
                "ai_generated_probability": ai_prob,
                "real_probability": 1 - ai_prob
            })
        except Exception as e:
            results.append({
                "image": os.path.basename(img_path),
                "prediction": "ERROR",
                "confidence": 0,
                "ai_generated_probability": 0,
                "real_probability": 0
            })
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["image", "prediction", "confidence", "ai_generated_probability", "real_probability"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Batch inference complete. Results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Predict if image is real or AI-generated')
    parser.add_argument('--image_path', type=str, default=None, help='Path to image file')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_dir', type=str, default=None, help='Directory of images for batch inference')
    args = parser.parse_args()

    config = Config()
    config.DEVICE = args.device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return

    if args.batch_dir:
        print("Loading model...")
        model = load_model(args.model_path, config)
        model = model.to(device)
        print(f"Running batch inference on {args.batch_dir}...")
        batch_inference(model, args.batch_dir, config, device)
        return

    if not args.image_path:
        print("Please provide --image_path or --batch_dir.")
        return
    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        return
    print("Loading model...")
    model = load_model(args.model_path, config)
    model = model.to(device)
    print("Preprocessing image...")
    image_tensor = preprocess_image(args.image_path, config)
    print("Making prediction...")
    prediction, confidence, ai_generated_prob = predict_image(model, image_tensor, device)
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Image: {args.image_path}")
    print(f"Prediction: {'AI-Generated' if prediction == 1 else 'Real'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"AI-Generated Probability: {ai_generated_prob:.2%}")
    print(f"Real Probability: {1 - ai_generated_prob:.2%}")
    print("="*50)
    if prediction == 1:
        print("This image appears to be AI-generated.")
    else:
        print("This image appears to be real.")
    if confidence > 0.9:
        print("High confidence prediction.")
    elif confidence > 0.7:
        print("Moderate confidence prediction.")
    else:
        print("Low confidence prediction - consider manual review.")

if __name__ == '__main__':
    main() 