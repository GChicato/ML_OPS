# api.py
import io, os, pathlib
import boto3
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------- MINIO (S3) CONFIG ----------
S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
S3_BUCKET = os.getenv("MINIO_BUCKET", "models")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "best_resnet18.pt")
S3_CLASSES_KEY = os.getenv("S3_CLASSES_KEY", "classes.txt")
S3_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
S3_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

MODEL_LOCAL = pathlib.Path("best_resnet18.pt")
CLASSES_LOCAL = pathlib.Path("classes.txt")

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS,
        aws_secret_access_key=S3_SECRET,
    )

def ensure_files():
    s3 = s3_client()
    if not MODEL_LOCAL.exists():
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(MODEL_LOCAL))
    if not CLASSES_LOCAL.exists():
        s3.download_file(S3_BUCKET, S3_CLASSES_KEY, str(CLASSES_LOCAL))

ensure_files()

# ---------- classes ----------
# classes.txt est de la forme: "0\tangry", "1\thappy", etc.
with open(CLASSES_LOCAL) as f:
    classes = [line.strip().split("\t")[-1] for line in f if line.strip()]
num_classes = len(classes)

# ---------- modèle ----------
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
state = torch.load(MODEL_LOCAL, map_location=DEVICE)  # <- charge le state_dict
model.load_state_dict(state, strict=True)              # <- on applique les poids
model.to(DEVICE)
model.eval()

# ---------- préprocessing ----------
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------- API ----------
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": DEVICE, "classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    return {"label": classes[idx.item()], "confidence": float(conf.item())}
