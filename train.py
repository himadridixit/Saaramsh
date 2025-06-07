import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import glob
import re


# --------------------------------------------------
# (1) DATASET & MODEL CLASSES
# --------------------------------------------------

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root = os.path.join(root_dir, split)
        self.classes = ['negative', 'positive']
        self.transform = transform
        self.samples = []
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.png'):
                    self.samples.append((os.path.join(class_dir, file), label_idx)) # file path and label as a tuple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

class CricketNet(nn.Module):
    def __init__(self):
        super(CricketNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),    # 64x64

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),    # 32x32

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),    # 16x16

            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten from 4D to 2D (batch_size, features)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 2. FOCAL LOSS IMPLEMENTATION
# --------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification:
     L = - α * (1 - p_t)^γ * log(p_t)
    where p_t is the model’s estimated probability for the true class.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction='mean'):
        """
        alpha: weight for the positive class (if you want to up‐weight positives, set alpha > 1)
        gamma: focusing parameter. Higher gamma → more focus on hard examples.
        reduction: 'mean' or 'sum' 
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: raw output from the network (shape: [B, 1])
        targets: ground‐truth labels (0 or 1, shape: [B, 1] or [B])
        """
        # Ensure targets has shape [B, 1]
        targets = targets.view(-1, 1).float()
        probs = torch.sigmoid(logits)         # shape [B,1], in (0,1)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Avoid log(0)
        eps = 1e-8
        log_p_t = torch.log(p_t.clamp(min=eps))

        loss = -alpha_t * ((1 - p_t) ** self.gamma) * log_p_t
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
        

# --------------------------------------------------
# (3) TRAIN/EVAL FUNCTION DEFINITIONS
# --------------------------------------------------

def get_sampler(dataset):
    class_counts = [0, 0]
    for _, label in dataset.samples:
        class_counts[label] += 1
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    # Wrap the DataLoader with tqdm to show batch‐level progress
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Wrap with tqdm if you want to track validation progress, too
    for inputs, labels in tqdm(loader, desc="Validation", leave=False):
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-10)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, recall, precision, f1, (tn, fp, fn, tp)


def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Recall plot
    plt.subplot(1, 2, 2)
    plt.plot(history['val_recall'], label='Val Recall', color='green')
    plt.title('Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def visualize_predictions(model, loader, device, num_samples=5):
    model.eval()
    classes = ['Negative', 'Positive']
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        idx = np.random.randint(0, len(loader.dataset))
        image, label = loader.dataset[idx]
        with torch.no_grad():
            inp = image.unsqueeze(0).to(device)
            out = model(inp)
            proba = torch.sigmoid(out).item()
            pred = 1 if proba > 0.5 else 0
        img_np = image.squeeze().cpu().numpy()
        axes[0, i].imshow(img_np, cmap='viridis')
        axes[0, i].set_title(f"True: {classes[label]}\nPred: {classes[pred]} ({proba:.2f})")
        axes[0, i].axis('off')
        # (You can add Grad‐CAM here if you like)
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

# --------------------------------------------------
# (4) MAIN TRAINING LOOP
# --------------------------------------------------

def find_latest_checkpoint(pattern="checkpoint_*.pth"):
    files = glob.glob(pattern)
    if not files:
        return None

    # Extract epoch numbers from filenames like "checkpoint_5.pth"
    def epoch_from_filename(fn):
        m = re.search(r"checkpoint_(\d+)\.pth$", fn)
        return int(m.group(1)) if m else -1

    files.sort(key=lambda fn: epoch_from_filename(fn))
    return files[-1]  # the one with the highest epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint.pth if it exists')
    args = parser.parse_args()
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 3. Datasets
    root_dir = 'dataset'
    train_dataset = SpectrogramDataset(root_dir, split='train', transform=train_transform)
    val_dataset   = SpectrogramDataset(root_dir, split='val',   transform=test_transform)
    test_dataset  = SpectrogramDataset(root_dir, split='test',  transform=test_transform)

    # 4. Sampler for class imbalance
    train_sampler = get_sampler(train_dataset)

    # 5. DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    # 6. Model, Loss, Optimizer
    model = CricketNet().to(device)
    # Compute pos_weight for FocalLoss’s α term if needed
    # For FocalLoss, α is just a scalar weight for positives.
    counts = np.bincount([label for _, label in train_dataset.samples])
    count_pos = counts[1]
    count_neg = counts[0]
    # If positives are rarer, set alpha > 1.0. For instance:
    # alpha = (count_neg + count_pos) / (2 * count_pos)
    alpha = 0.5
    gamma = 2.0

    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler monitors “max” of validation F1
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 7. Training Loop
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = None
    
    start_epoch = 0
    if args.resume:
        latest_ckpt = find_latest_checkpoint()
        if latest_ckpt is not None:
            print(f"=> Loading checkpoint '{latest_ckpt}'")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            start_epoch = ckpt['epoch'] + 1
            best_val_f1 = ckpt['best_val_f1']
            patience_counter = ckpt['patience_counter']

            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"=> Resumed from epoch {ckpt['epoch']}, best_val_f1={best_val_f1:.4f}")
        else:
            print("=> No checkpoint found, training from scratch.")

    history = {'train_loss': [], 'val_loss': [], 'val_recall': []}
    num_epochs = 30
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_recall, val_precision, val_f1, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_recall'].append(val_recall)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val Precision: {val_precision:.4f} | "
              f"Val Recall: {val_recall:.4f}")

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            # Save as checkpoint_{epoch+1}.pth
            os.makedirs('checkpoints', exist_ok=True)
            filename = f'checkpoints/checkpoint_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, filename)
            print(f"  → New best (F1={val_f1:.4f}), Loss={val_loss:.4f}); saved as {filename}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    # 8. Load best model & evaluate on test set
    if best_epoch is not None:
        best_path = f'checkpoints/checkpoint_{best_epoch}.pth'
        print(f"=> Loading best‐model checkpoint from epoch {best_epoch}")
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        print("No checkpoint was ever saved—cannot load best model.")
    test_loss, test_recall, test_precision, test_f1, (tn, fp, fn, tp) = evaluate(model, test_loader, criterion, device)
    print("\n" + "="*50)
    print(f"Test Performance (using best model):")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print("  Confusion Matrix:")
    print(f"    TP: {tp} | FP: {fp}")
    print(f"    FN: {fn} | TN: {tn}")
    print("="*50)

    # 9. Plot & save metrics
    plot_metrics(history)

    # 10. Sample predictions
    visualize_predictions(model, test_loader, device)

# --------------------------------------------------
# 4. ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()    
    main()
