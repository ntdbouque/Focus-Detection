import torch
from facenet_pytorch import MTCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Thiết lập MTCNN
mtcnn = MTCNN(image_size=160, margin=14, min_face_size=20)

def process_images(dataloader):
    """Hàm xử lý ảnh đầu vào để trích xuất khuôn mặt."""
    for i, (img, label) in enumerate(dataloader):
        faces = []
        for im in img:
            face = mtcnn(im)
            if face is not None:
                faces.append(face)
        
        if len(faces) > 0:
            faces = torch.stack(faces)
            torch.save(faces, f"processed_faces_batch_{i}.pt")
        
        print(f"Đã xử lý batch {i+1}/{len(dataloader)}")

# Dataset và DataLoader
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='path_to_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Chạy xử lý ảnh
process_images(dataloader)
