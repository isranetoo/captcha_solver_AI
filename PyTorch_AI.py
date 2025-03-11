import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
from PIL import Image

# Definição do dataset personalizado
class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None, captcha_length=4, filenames=None):
        self.data_dir = data_dir
        self.transform = transform
        # Check if the directory exists
        if not os.path.exists(data_dir):
            print(f"Warning: Dataset directory '{data_dir}' does not exist.")
            self.filenames = []
        else:
            if filenames is None:
                self.filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            else:
                self.filenames = filenames
                
            print(f"Found {len(self.filenames)} files in dataset")
            if len(self.filenames) > 0:
                print(f"Sample filenames: {self.filenames[:3]}")
            
        self.characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}
        self.captcha_length = captcha_length

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        label_text = self.filenames[idx].split("_")[0]  # Nome do arquivo: "1234_xyz.png"
        label_text = ''.join([c for c in label_text if c in self.characters])  # Filter invalid characters
        
        # Load and process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 50)) / 255.0

        # Convert to tensor with proper shape [channels, height, width]
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension at position 0
        
        # Pad or truncate label to fixed length
        if len(label_text) > self.captcha_length:
            label_text = label_text[:self.captcha_length]
        elif len(label_text) < self.captcha_length:
            label_text = label_text + '0' * (self.captcha_length - len(label_text))
        
        # Convert characters to indices
        label = [self.char_to_index[c] for c in label_text]

        if self.transform:
            # Convert to PIL image for torchvision transforms
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            img_tensor = self.transform(pil_img)

        return img_tensor, torch.tensor(label, dtype=torch.long)

# Dataset e DataLoader
# Use absolute path to avoid relative path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "dataset", "captcha_images")
print(f"Looking for dataset at: {data_dir}")

# Alternative paths to try if the default one doesn't exist
alternative_paths = [
    os.path.join(current_dir, "captcha_images"),
    "C:/Users/IsraelAntunes/Desktop/captcha_solver_AI/dataset/captcha_images",
    "C:/Users/IsraelAntunes/Desktop/captcha_solver_AI/captcha_images"
]

if not os.path.exists(data_dir):
    for alt_path in alternative_paths:
        print(f"Trying alternative path: {alt_path}")
        if os.path.exists(alt_path):
            data_dir = alt_path
            print(f"Using alternative path: {data_dir}")
            break
    else:
        print("Warning: Could not find dataset directory. Please create it and add images.")
        # Create a minimal directory structure for testing
        os.makedirs(data_dir, exist_ok=True)

# Data Augmentation - Add transforms for better generalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),  # Rotação de até 10 graus
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Distorção
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translação
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variações de brilho e contraste
])

captcha_length = 4
dataset = CaptchaDataset(data_dir, transform=transform, captcha_length=captcha_length)

# Check if dataset is empty
if len(dataset) == 0:
    print("No images found in the dataset directory. Please add images before training.")
    import sys
    sys.exit(1)

dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

# Split the dataset into training and testing sets
if len(dataset.filenames) > 0:
    train_files, test_files = train_test_split(dataset.filenames, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_files)} images, Test set: {len(test_files)} images")
    
    train_dataset = CaptchaDataset(data_dir, transform=transform, captcha_length=captcha_length, filenames=train_files)
    test_dataset = CaptchaDataset(data_dir, transform=transform, captcha_length=captcha_length, filenames=test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(32, len(test_dataset)), shuffle=False)
else:
    print("No images found. Cannot split dataset.")
    train_loader = dataloader
    test_loader = dataloader

# Criando modelo com PyTorch para reconhecimento de múltiplos caracteres - Modelo melhorado
class CaptchaSolver(nn.Module):
    def __init__(self, num_chars=4, num_classes=36):
        super(CaptchaSolver, self).__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        # Enhanced Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Added BatchNorm for better training stability
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Nova camada
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),  # Additional dropout for regularization
            
            nn.Flatten()
        )
        
        # Calculate the output size after convolutions and flattening
        self.conv_output_size = self._get_conv_output_size((1, 50, 100))
        
        # Multiple heads for each character position with increased dropout
        self.char_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.conv_output_size, 256),  # Larger hidden layer
                nn.ReLU(),
                nn.Dropout(0.3),  # Increased dropout rate
                nn.Linear(256, 128),  # Adding a second hidden layer
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            ) for _ in range(num_chars)
        ])

    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size of the convolutional layers
        batch_size = 1
        input_tensor = torch.zeros(batch_size, *shape)
        output = self.conv_layers(input_tensor)
        return int(np.prod(output.size()[1:]))
        
    def forward(self, x):
        # Extract features from convolutional layers
        features = self.conv_layers(x)
        
        # Apply each character predictor
        outputs = [predictor(features) for predictor in self.char_predictors]
        
        # Stack outputs along a new dimension
        return torch.stack(outputs, dim=1)

# Alternative CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_chars=4, num_classes=36):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 50x100 -> 25x50
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 25x50 -> 12x25
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 12x25 -> 6x25
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Recurrent layers
        self.rnn = nn.LSTM(256 * 6, 256, bidirectional=True, batch_first=True)
        
        # Predictor for each character
        self.char_predictors = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_chars)
        ])
        
        self.num_chars = num_chars
    
    def forward(self, x):
        # Extract features with CNN
        conv_output = self.conv_layers(x)
        
        # Reshape for RNN: batch_size x sequence_length x features
        batch_size = conv_output.size(0)
        conv_output = conv_output.permute(0, 3, 1, 2)  # B x W x C x H
        conv_output = conv_output.reshape(batch_size, conv_output.size(1), -1)
        
        # RNN layers
        rnn_output, _ = self.rnn(conv_output)
        
        # Pool across the width dimension to get a single feature vector
        features = torch.mean(rnn_output, dim=1)
        
        # Predict each character
        char_outputs = [predictor(features) for predictor in self.char_predictors]
        return torch.stack(char_outputs, dim=1)

# Function to generate synthetic CAPTCHAs
def generate_synthetic_captchas(output_dir, num_samples=1000):
    try:
        from captcha.image import ImageCaptcha
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        image_captcha = ImageCaptcha(width=100, height=50)
        
        print(f"Generating {num_samples} synthetic CAPTCHAs...")
        for i in range(num_samples):
            text = ''.join(random.choices(characters, k=4))
            image_captcha.write(text, os.path.join(output_dir, f"{text}_{i}.png"))
            
            if i % 100 == 0:
                print(f"Generated {i} CAPTCHAs...")
                
        print(f"Completed generating {num_samples} CAPTCHAs in {output_dir}")
    except ImportError:
        print("Captcha generator not installed. Install with: pip install captcha")
        print("Skipping synthetic CAPTCHA generation.")

# Uncomment to generate synthetic data if needed
# synthetic_dir = os.path.join(current_dir, "dataset", "synthetic_captchas")
# generate_synthetic_captchas(synthetic_dir, num_samples=5000)

# Treinando o modelo
# Use model = CRNN(num_chars=captcha_length, num_classes=len(dataset.characters)) for CRNN model instead
model = CaptchaSolver(num_chars=captcha_length, num_classes=len(dataset.characters))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
early_stop_counter = 0

num_epochs = 150  # Increased number of epochs
for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(imgs)
        
        # Calculate accuracy
        _, preds = outputs.max(2)  # Get predictions for each character
        correct_chars += (preds == labels).sum().item()
        total_chars += labels.numel()
        
        # Reshape for loss calculation: [batch, num_chars, num_classes] -> [batch * num_chars, num_classes]
        batch_size = outputs.size(0)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    train_accuracy = correct_chars / total_chars
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0
    val_correct_chars = 0
    val_total_chars = 0
    val_correct_captchas = 0
    total_captchas = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            
            # Calculate validation loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            batch_loss = criterion(outputs_flat, labels_flat)
            val_loss += batch_loss.item()
            
            # Calculate character accuracy
            _, preds = outputs.max(2)
            val_correct_chars += (preds == labels).sum().item()
            val_total_chars += labels.numel()
            
            # Calculate whole captcha accuracy
            captcha_correct = torch.all(preds == labels, dim=1).sum().item()
            val_correct_captchas += captcha_correct
            total_captchas += labels.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    val_char_accuracy = val_correct_chars / val_total_chars
    val_captcha_accuracy = val_correct_captchas / total_captchas
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Check for overfitting
    overfitting_msg = ""
    if train_accuracy > 0.9 and val_char_accuracy < 0.7:
        overfitting_msg = " (Possível overfitting detectado!)"
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    print(f'Train Char Accuracy: {train_accuracy:.4f}, Val Char Accuracy: {val_char_accuracy:.4f}')
    print(f'Val Whole CAPTCHA Accuracy: {val_captcha_accuracy:.4f}{overfitting_msg}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "captcha_model_best.pth")
        print("Saved best model checkpoint")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        
    if early_stop_counter >= patience:
        print("Early stopping triggered!")
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load("captcha_model_best.pth"))

# Save the final model
torch.save(model.state_dict(), "captcha_model_final.pth")

# Detailed evaluation of the model
model.eval()
test_correct_chars = 0
test_total_chars = 0
test_correct_captchas = 0
test_total_captchas = 0

char_correct_count = {}  # Track accuracy per character position
char_total_count = {}

confusion_matrix = torch.zeros(len(dataset.characters), len(dataset.characters), dtype=torch.int)

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, preds = outputs.max(2)
        
        # Character-level accuracy
        test_correct_chars += (preds == labels).sum().item()
        test_total_chars += labels.numel()
        
        # CAPTCHA-level accuracy
        captcha_correct = torch.all(preds == labels, dim=1)
        test_correct_captchas += captcha_correct.sum().item()
        test_total_captchas += labels.size(0)
        
        # Per-position accuracy
        for pos in range(captcha_length):
            pos_correct = (preds[:, pos] == labels[:, pos]).sum().item()
            pos_total = labels.size(0)
            
            if pos not in char_correct_count:
                char_correct_count[pos] = 0
                char_total_count[pos] = 0
                
            char_correct_count[pos] += pos_correct
            char_total_count[pos] += pos_total
        
        # Build confusion matrix
        for i in range(labels.size(0)):
            for j in range(captcha_length):
                true_idx = labels[i, j]
                pred_idx = preds[i, j]
                confusion_matrix[true_idx, pred_idx] += 1

char_accuracy = test_correct_chars / test_total_chars
captcha_accuracy = test_correct_captchas / test_total_captchas

print(f"\nFinal Evaluation Results:")
print(f"Character-level accuracy: {char_accuracy:.4f}")
print(f"Complete CAPTCHA accuracy: {captcha_accuracy:.4f}")

print("\nAccuracy by character position:")
for pos in range(captcha_length):
    pos_acc = char_correct_count[pos] / char_total_count[pos]
    print(f"Position {pos+1}: {pos_acc:.4f}")

# Testando o modelo com função melhorada
def predict_captcha(model, img_path, char_map):
    try:
        # Process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        img = cv2.resize(img, (100, 50)) / 255.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Get predictions
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=2)  # Add softmax to get probabilities
            confidences, preds = probs.max(2)
        
        # Convert indices to characters
        preds = preds.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        chars = [char_map[idx] for idx in preds]
        captcha_text = ''.join(chars)
        
        # Return prediction and confidence
        return captcha_text, confidences
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

# Create character map (index -> character)
idx_to_char = {idx: char for char, idx in dataset.char_to_index.items()}

# Try to use test_captcha.png, if not available, use a sample from the dataset
test_image = os.path.join(current_dir, "test_captcha.png")
print(f"Looking for test image at: {test_image}")

try:
    predicted_text, confidences = predict_captcha(model, test_image, idx_to_char)
    if predicted_text:
        print(f"Predicted CAPTCHA from {test_image}: {predicted_text}")
        print(f"Confidence levels: {confidences}")
    else:
        raise ValueError("Failed to predict from test file")
except Exception as e:
    print(f"Falling back to a sample from the dataset: {e}")
    # Use a sample from the dataset instead
    if len(dataset.filenames) > 0:
        sample_img_path = os.path.join(dataset.data_dir, dataset.filenames[0])
        print(f"Testing with sample image: {sample_img_path}")
        
        # Check if the file exists
        if not os.path.exists(sample_img_path):
            print(f"Error: Sample image file does not exist: {sample_img_path}")
        else:
            actual_text = dataset.filenames[0].split("_")[0]
            actual_text = ''.join([c for c in actual_text if c in dataset.characters])
            predicted_text, confidences = predict_captcha(model, sample_img_path, idx_to_char)
            if predicted_text:
                print(f"Sample image: {dataset.filenames[0]}")
                print(f"Actual text: {actual_text}")
                print(f"Predicted text: {predicted_text}")
                print(f"Character confidences: {confidences}")
            else:
                print("Failed to predict from sample image as well.")
    else:
        print("No images available in the dataset.")
