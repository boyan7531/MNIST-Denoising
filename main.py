import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset,batch_size=512,shuffle=True)

# Define a simple AE
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

# Train the model
model = DenoisingAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(20):
    loss = None
    for clean_images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        clean_images = clean_images.to(device)

        noise = 0.5 * torch.randn_like(clean_images)
        noisy_images = torch.clamp(clean_images + noise, 0., 1.)


        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")

# Visualising the result
import matplotlib.pyplot as plt
clean_images, _ = next(iter(dataloader))
clean_images = clean_images.to(device)

noise = 0.5 * torch.randn_like(clean_images)
noisy_images = torch.clamp(clean_images + noise, 0., 1.)

outputs = model(noisy_images) # Get denoised output

for i in range(5):
    plt.subplot(3, 5, i + 1)
    # Show original clean image
    plt.imshow(clean_images[i].cpu().squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3, 5, i + 6)
    # Show noisy input image
    plt.imshow(noisy_images[i].cpu().squeeze(), cmap='gray')
    plt.title('Noisy Input')
    plt.axis('off')

    plt.subplot(3, 5, i + 11) # Adjust subplot index for the third row
    # Show denoised output image
    plt.imshow(outputs[i].detach().cpu().squeeze(), cmap='gray')
    plt.title('Denoised')
    plt.axis('off')

plt.show()
