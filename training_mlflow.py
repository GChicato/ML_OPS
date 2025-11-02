from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import mlflow

DATA_DIR = Path("data")
ART = Path("artifacts")
MODEL_WEIGHTS = ART/"best_resnet18.pt"
CLASSES_TXT   = ART/"classes.txt"

IMG=224
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# charge classes
classes = [line.strip().split("\t")[1] for line in open(CLASSES_TXT)]
n = len(classes)

# dataloader test
tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_ds = datasets.ImageFolder(DATA_DIR/"test", transform=tf)
test_ld = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

# modèle
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, n)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.to(DEVICE); model.eval()

@torch.no_grad()
def eval_accuracy():
    total = correct = 0
    for x,y in test_ld:
        x,y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        correct += (p==y).sum().item()
        total   += x.size(0)
    return correct/total

def main():
    mlflow.set_experiment("emotions")
    with mlflow.start_run(run_name="log-existing-model"):
        # ici tu logues ce que tu connais déjà (params simples)
        mlflow.log_params({"backbone":"resnet18","img":IMG,"device":DEVICE,"trained_offline":True})
        # métrique test (recalculée)
        test_acc = eval_accuracy()
        mlflow.log_metric("test_acc", test_acc)
        # artefacts (poids + classes)
        mlflow.log_artifact(str(MODEL_WEIGHTS))
        mlflow.log_artifact(str(CLASSES_TXT))
        print("Logged to MLflow. TEST acc:", test_acc)

if __name__ == "__main__":
    main()
