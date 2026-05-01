import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DefectClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "../data/bottle"
BATCH_SIZE  = 16
EPOCHS      = 15
LR          = 0.001
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── DATASET ───────────────────────────────────────────────────────────────────
class BottleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # Train folder only has "good" images
        train_good = os.path.join(data_dir, "train", "good")
        for img in os.listdir(train_good):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append((os.path.join(train_good, img), 0))  # 0 = good
        
        # Test folder has good and defect categories
        test_dir = os.path.join(data_dir, "test")
        for category in os.listdir(test_dir):
            cat_path = os.path.join(test_dir, category)
            if not os.path.isdir(cat_path):
                continue
            label = 0 if category == "good" else 1  # 1 = defect
            for img in os.listdir(cat_path):
                if img.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cat_path, img), label))
        
        print(f"Total samples: {len(self.samples)}")
        good_count = sum(1 for _, l in self.samples if l == 0)
        defect_count = sum(1 for _, l in self.samples if l == 1)
        print(f"Good: {good_count} | Defect: {defect_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
full_dataset = BottleDataset(DATA_DIR, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── MODEL ─────────────────────────────────────────────────────────────────────
model     = DefectClassifier(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
train_losses, val_losses, val_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_train = running_loss / len(train_loader)
    avg_val   = val_loss / len(val_loader)
    acc       = 100 * correct / total
    
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    val_accs.append(acc)
    
    scheduler.step()
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f} | Val Acc={acc:.2f}%")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/defect_classifier.pth")
print("Model saved to models/defect_classifier.pth")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses,   label="Val Loss")
ax1.set_title("Loss Curves"); ax1.legend()
ax2.plot(val_accs, label="Val Accuracy", color="green")
ax2.set_title("Validation Accuracy"); ax2.legend()
plt.tight_layout()
plt.savefig("../training_curves.png")
plt.show()
print("Training complete.")