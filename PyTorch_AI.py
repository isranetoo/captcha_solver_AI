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
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = os.listdir(data_dir)
        self.characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        label_text = self.filenames[idx].split("_")[0]  # Nome do arquivo: "1234_xyz.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 50)) / 255.0
        img = np.expand_dims(img, axis=0)  # Adicionando canal

        label = [self.char_to_index[c] for c in label_text]

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Dataset e DataLoader
data_dir = "dataset/captcha_images/"
transform = transforms.ToTensor()
dataset = CaptchaDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Criando modelo com PyTorch
class CaptchaSolver(nn.Module):
    def __init__(self):
        super(CaptchaSolver, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 36),  # 36 classes (A-Z, 0-9)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Treinando o modelo
model = CaptchaSolver()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # Número de épocas
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels.view(-1))  # Ajustando rótulo para loss function
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Testando o modelo
def predict_captcha(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 50)) / 255.0
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction.argmax(dim=1).item()

test_image = "test_captcha.png"
print("Predicted CAPTCHA:", predict_captcha(test_image))
