from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = Path("data")
EPOCHS = 10
BATCH = 32
LR = 1e-3
IMG = 224
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def loaders():
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.ImageFolder(DATA_DIR/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_DIR/"val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(DATA_DIR/"test",  transform=val_tf)
    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2),
        train_ds.classes
    )

def run_epoch(model, loader, criterion, opt=None, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc="train" if train else "eval"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train: opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main():
    train_ld, val_ld, test_ld, classes = loaders()
    n = len(classes)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n)
    model.to(DEVICE)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = 0.0
    Path("artifacts").mkdir(exist_ok=True)

    for e in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, train_ld, crit, opt, train=True)
        va_loss, va_acc = run_epoch(model, val_ld,   crit, None, train=False)
        print(f"[{e}/{EPOCHS}] train acc {tr_acc:.3f} | val acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), "artifacts/best_resnet18.pt")
            with open("artifacts/classes.txt", "w") as f:
                for i, c in enumerate(classes): f.write(f"{i}\t{c}\n")

    te_loss, te_acc = run_epoch(model, test_ld, crit, None, train=False)
    print(f"TEST acc: {te_acc:.3f}")

if __name__ == "__main__":
    main()
