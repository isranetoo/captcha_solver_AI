import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import argparse
from PyTorch_AI import CaptchaSolver, CRNN, predict_captcha

def load_model(model_path, model_type="captcha_solver", num_chars=4, num_classes=36):
    """
    Load a trained CAPTCHA solver model
    
    Args:
        model_path: Path to the saved model file
        model_type: Type of model ('captcha_solver' or 'crnn')
        num_chars: Number of characters in the CAPTCHA
        num_classes: Number of possible classes (characters)
        
    Returns:
        The loaded model in evaluation mode
    """
    if model_type.lower() == "captcha_solver":
        model = CaptchaSolver(num_chars=num_chars, num_classes=num_classes)
    elif model_type.lower() == "crnn":
        model = CRNN(num_chars=num_chars, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model

def create_char_map(characters="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Create a mapping from indices to characters"""
    return {idx: char for idx, char in enumerate(characters)}

def main():
    parser = argparse.ArgumentParser(description="CAPTCHA Prediction Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to the CAPTCHA image")
    parser.add_argument("--model", type=str, default="captcha_model_best.pth", 
                        help="Path to the model file (default: captcha_model_best.pth)")
    parser.add_argument("--model_type", type=str, default="captcha_solver", 
                        choices=["captcha_solver", "crnn"], help="Type of model architecture")
    parser.add_argument("--chars", type=int, default=4, help="Number of characters in the CAPTCHA")
    parser.add_argument("--classes", type=int, default=36, 
                        help="Number of character classes (default: 36 for 0-9 and A-Z)")
    
    args = parser.parse_args()
    
    # Create character map
    char_map = create_char_map()
    
    # Load model
    try:
        model = load_model(
            args.model, 
            model_type=args.model_type,
            num_chars=args.chars,
            num_classes=args.classes
        )
        print(f"Model loaded successfully from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    try:
        predicted_text, confidences = predict_captcha(model, args.image, char_map)
        if predicted_text:
            print(f"Predicted CAPTCHA text: {predicted_text}")
            print(f"Character confidences: {confidences}")
            
            # Print per-character confidence
            for idx, (char, conf) in enumerate(zip(predicted_text, confidences)):
                print(f"Character {idx+1}: '{char}' with confidence {conf:.4f}")
        else:
            print("Failed to predict CAPTCHA")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
