import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()

model.train()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='path_to_faces', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            embeddings = model(images)
            outputs = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

train_model()