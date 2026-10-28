import os
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.optim import Adam
from collections import defaultdict

# Reproducibility
random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load training data
train_dataset = datasets.ImageFolder(
    "train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "test",
    transform=transform
)

print("Classes:", train_dataset.classes)

# Select a random, balanced training subset
indices_by_class = defaultdict(list)

for index, (_, label) in enumerate(train_dataset.samples):
    indices_by_class[label].append(index)

selected_indices = []

for label, indices in indices_by_class.items():
    random.shuffle(indices)
    selected_indices.extend(indices[:500])

print("Selected training images:", len(selected_indices))

small_train_dataset = Subset(
    train_dataset,
    selected_indices
)

train_loader = DataLoader(
    small_train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# Load pretrained MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze all pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final feature block
for parameter in model.features[-1].parameters():
    parameter.requires_grad = True

# Replace the classification layer
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(train_dataset.classes)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)

# Train
for epoch in range(5):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/5, "
        f"Loss: {average_loss:.4f}"
    )

# Save the trained model
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    "models/plastic_model.pth"
)

print("Improved model saved successfully.")