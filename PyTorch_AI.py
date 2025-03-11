import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np

# Definição do dataset personalizado
class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None, captcha_length=4):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = os.listdir(data_dir)
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
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label, dtype=torch.long)

# Dataset e DataLoader
data_dir = "dataset/captcha_images/"
captcha_length = 4
transform = None
dataset = CaptchaDataset(data_dir, transform=transform, captcha_length=captcha_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Criando modelo com PyTorch para reconhecimento de múltiplos caracteres
class CaptchaSolver(nn.Module):
    def __init__(self, num_chars=4, num_classes=36):
        super(CaptchaSolver, self).__init__()
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate the output size after convolutions and flattening
        self.conv_output_size = self._get_conv_output_size((1, 50, 100))
        
        # Multiple heads for each character position
        self.char_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.conv_output_size, 128),
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

# Treinando o modelo
model = CaptchaSolver(num_chars=captcha_length, num_classes=len(dataset.characters))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(imgs)
        
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
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Save the model
torch.save(model.state_dict(), "captcha_model.pth")

# Testando o modelo
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
            _, preds = outputs.max(2)
        
        # Convert indices to characters
        preds = preds.cpu().numpy()[0]
        chars = [char_map[idx] for idx in preds]
        captcha_text = ''.join(chars)
        
        return captcha_text
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Create character map (index -> character)
idx_to_char = {idx: char for char, idx in dataset.char_to_index.items()}

# Try to use test_captcha.png, if not available, use a sample from the dataset
test_image = "test_captcha.png"
try:
    predicted_text = predict_captcha(model, test_image, idx_to_char)
    if predicted_text:
        print(f"Predicted CAPTCHA from {test_image}: {predicted_text}")
    else:
        raise ValueError("Failed to predict from test file")
except Exception as e:
    print(f"Falling back to a sample from the dataset: {e}")
    # Use a sample from the dataset instead
    if len(dataset.filenames) > 0:
        sample_img_path = os.path.join(dataset.data_dir, dataset.filenames[0])
        actual_text = dataset.filenames[0].split("_")[0]
        actual_text = ''.join([c for c in actual_text if c in dataset.characters])
        predicted_text = predict_captcha(model, sample_img_path, idx_to_char)
        if predicted_text:
            print(f"Sample image: {dataset.filenames[0]}")
            print(f"Actual text: {actual_text}")
            print(f"Predicted text: {predicted_text}")
        else:
            print("Failed to predict from sample image as well.")
    else:
        print("No images available in the dataset.")
