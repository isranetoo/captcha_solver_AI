import os
import pandas as pd
import argparse
from predict_captcha import load_model, create_char_map
from PyTorch_AI import predict_captcha

def process_directory(model, image_dir, char_map, output_file=None):
    """
    Process all images in a directory and save predictions to a CSV file
    
    Args:
        model: The loaded CAPTCHA solver model
        image_dir: Directory containing CAPTCHA images
        char_map: Mapping from indices to characters
        output_file: Path to save results (optional)
        
    Returns:
        DataFrame with predictions
    """
    results = []
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(image_files)
    
    print(f"Found {total_files} images to process")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{total_files}...")
            
        img_path = os.path.join(image_dir, img_file)
        try:
            predicted_text, confidences = predict_captcha(model, img_path, char_map)
            
            if predicted_text:
                results.append({
                    'image': img_file,
                    'prediction': predicted_text,
                    'confidence': sum(confidences) / len(confidences)  # Average confidence
                })
            else:
                print(f"Failed to predict {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to file if specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Batch CAPTCHA Prediction Tool")
    parser.add_argument("--dir", type=str, required=True, help="Directory with CAPTCHA images")
    parser.add_argument("--model", type=str, default="captcha_model_best.pth", 
                        help="Path to the model file")
    parser.add_argument("--model_type", type=str, default="captcha_solver", 
                        choices=["captcha_solver", "crnn"], help="Type of model architecture")
    parser.add_argument("--output", type=str, default="predictions.csv", 
                        help="Output CSV file path")
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
    
    # Process directory
    process_directory(model, args.dir, char_map, args.output)

if __name__ == "__main__":
    main()
