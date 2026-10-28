import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

# 1. Select device
device = torch.device("cpu")

# 2. Prepare image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Load the test dataset
test_dataset = datasets.ImageFolder(
    "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# 4. Rebuild the trained MobileNetV2 model
checkpoint = torch.load(
    "models/plastic_model.pth",
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = models.mobilenet_v2(weights=None)

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# 5. Evaluate predictions
correct = 0
total = 0

confusion_matrix = torch.zeros(
    len(classes),
    len(classes),
    dtype=torch.int64
)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for actual, predicted in zip(labels, predictions):
            confusion_matrix[actual, predicted] += 1

# 6. Display results
accuracy = 100 * correct / total

print("\nModel Evaluation")
print("-------------------------")
print("Classes:", classes)
print("Total test images:", total)
print(f"Test accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix")
print("Rows = Actual classes")
print("Columns = Predicted classes")
print(confusion_matrix)

print("\nPer-class results")

for i, class_name in enumerate(classes):
    true_positive = confusion_matrix[i, i].item()
    actual_count = confusion_matrix[i].sum().item()
    predicted_count = confusion_matrix[:, i].sum().item()

    recall = (
        true_positive / actual_count
        if actual_count else 0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count else 0
    )

    print(
        f"{class_name}: "
        f"Precision = {precision:.3f}, "
        f"Recall = {recall:.3f}"
    )