import torch
import torch.nn as nn
import torch.optim as optim
import os # Added for directory creation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# --- Define the Model Architecture (must match the saved model) ---
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding= 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# ------------------------------------------------------------------

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "denoising_ae_mnist.pth")
IMAGE_PATH = "mnist-3.png" # Make sure this image exists
NOISE_FACTOR = 0.5
# ---------------------

# --- Load Model ---
model = DenoisingAutoencoder().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (important!)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
# ------------------

# --- Load and Prepare Image ---
# Define the same transformation as used for training data (without noise)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Ensure image is grayscale
    transforms.ToTensor(),
])

try:
    img = Image.open(IMAGE_PATH)
except FileNotFoundError:
    print(f"Error: Image file not found at {IMAGE_PATH}")
    exit()

# Apply transformation (adds batch dimension and moves to device)
original_tensor = transform(img).unsqueeze(0).to(device) # unsqueeze adds batch dim [1, 1, 28, 28]

# Add noise
noise = NOISE_FACTOR * torch.randn_like(original_tensor)
noisy_tensor = torch.clamp(original_tensor + noise, 0., 1.)
# ----------------------------

# --- Denoise Image ---
with torch.no_grad(): # Deactivate autograd engine for inference
    denoised_tensor = model(noisy_tensor)
# --------------------

# --- Visualize ---
plt.figure(figsize=(9, 3))

# Original
plt.subplot(1, 3, 1)
plt.imshow(original_tensor.cpu().squeeze().numpy(), cmap='gray')
plt.title('Original')
plt.axis('off')

# Noisy
plt.subplot(1, 3, 2)
plt.imshow(noisy_tensor.cpu().squeeze().numpy(), cmap='gray')
plt.title('Noisy')
plt.axis('off')

# Denoised
plt.subplot(1, 3, 3)
plt.imshow(denoised_tensor.cpu().squeeze().numpy(), cmap='gray')
plt.title('Denoised')
plt.axis('off')

plt.tight_layout()
plt.show()
# ----------------


