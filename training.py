import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
import torch

from sklearn.model_selection import train_test_split

from collections import Counter


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])

])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
    ])


train_dataset = datasets.ImageFolder(
    root = r"D:\hyper-kvasir-labeled-images-extracted\labeled-images\lower-gi-tract",
    transform = train_transform
)

val_dataset = datasets.ImageFolder(
    root = r"D:\hyper-kvasir-labeled-images-extracted\labeled-images\lower-gi-tract",
    transform = val_transform
)



cecum_idx = train_dataset.class_to_idx['cecum']
all_labels = [train_dataset.samples[i][1] for i in range(len(train_dataset))]
binary_all_labels = [1 if label == cecum_idx else 0 for label in all_labels]


indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(
    indices,
    test_size = 0.2,
    stratify=binary_all_labels,
    random_state=42,
)

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

n_cecum = sum(binary_all_labels)
n_non_cecum = len(binary_all_labels) - n_cecum

binary_train_labels = [binary_all_labels[i] for i in train_indices]

weights = [1/n_cecum if label == 1 else 1/n_non_cecum for label in binary_train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)




model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)


#freeze all the layers
for param in model.parameters():
    param.requires_grad = False

#unfreeze the last 20 
for param in list(model.parameters())[-20:]:
    param.requires_grad = True

#replacing the final layer with a binary classifier instead of the original 1000 classes in ImageNet
model.fc = nn.Sequential(
    nn.Linear(2048,256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,1),
    
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

pos_weight = torch.tensor([n_non_cecum/n_cecum]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode = 'min', factor = 0.5, patience = 3
)

scaler = GradScaler()


def training_loop():
    num_epochs = 20
    best_val_loss = float('inf')
    train_loader = DataLoader(
     train_dataset,
     batch_size = 32,
     sampler = sampler,
     num_workers = 4,
     pin_memory = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = 32,
        shuffle = False,
        num_workers = 4,
        pin_memory = True
    )



    for epoch in range(num_epochs):
        #training

        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            binary = (labels == cecum_idx).float().to(device)
            
            optimizer.zero_grad()

            with autocast(device_type = device.type):
                outputs = model(images).squeeze()
                loss = criterion(outputs, binary)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        #validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        tp = 0 #true positives
        fn = 0 #false negatives

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                binary = (labels == cecum_idx).float().to(device)
                with autocast(device_type = device.type):
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, binary)

                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == binary).sum().item()
                total += binary.size(0) #batch size
                tp += ((preds == 1) & (binary == 1)).sum().item()
                fn += ((preds == 0) & (binary == 1)).sum().item()


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved at epoch {epoch +1}")

        accuracy = correct/total
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        scheduler.step(avg_val_loss)


        print(f"Epoch {epoch + 1}/{num_epochs} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
if __name__ == "__main__":
    training_loop()