🎭 Emotion Classifier – MLOps End-to-End

Pipeline complet : data → entraînement PyTorch → suivi MLflow → stockage MinIO → API FastAPI → UI Streamlit → Docker Compose.

📦 Stack Technique
Composant	Rôle
PyTorch (ResNet18)	Fine-tuning pour classifier les émotions
MLflow	Tracking des expériences, métriques, paramètres
MinIO (S3 local)	Stockage du modèle entraîné (best_resnet18.pt)
FastAPI	API d’inférence /predict
Streamlit	Interface simple pour tester le modèle
Docker Compose	Orchestration complète
✅ Fonctionnalités

Classification des émotions (7 classes)

Modèle fine-tuné ResNet18 + preprocessing

Chargement du modèle depuis MinIO au lancement de l’API

UI Streamlit conviviale (upload ou URL)

Architecture Dockerisée & reproductible

📁 Structure du Projet
mlops/
│── api/
│   ├── api.py
│   ├── requirements.txt
│   └── Dockerfile
│
│── app/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
│── minio_data/        # Volume MinIO (généré automatiquement)
│
│── artifacts/         # Modèle (.pt) + classes.txt après entraînement
│
│── training/          # Code d’entraînement (si ajouté)
│
│── docker-compose.yml
│── README.md
│── .gitignore

🚀 Lancement du Projet
✅ 1. Build + start tous les services
docker compose up -d --build


Cela lance :

✅ FastAPI → http://localhost:8080
✅ Streamlit → http://localhost:8501
✅ MinIO Console → http://localhost:9101
✅ MinIO S3 API → http://localhost:9100

📤 Upload du modèle dans MinIO (obligatoire avant l’inférence)

MinIO doit contenir :

models/
  ├── best_resnet18.pt
  └── classes.txt


Depuis ton terminal :

mc alias set myminio http://localhost:9100 minioadmin minioadmin123
mc mb myminio/models
mc cp artifacts/best_resnet18.pt myminio/models/best_resnet18.pt
mc cp artifacts/classes.txt myminio/models/classes.txt

🧪 Tester l’API FastAPI
POST  http://localhost:8080/predict
form-data: file=@image.jpg

🖥️ Streamlit UI

Accès :
👉 http://localhost:8501

Fonctionnalités :
✅ Upload d’image
✅ URL d’image
✅ Prévisualisation
✅ Envoi vers l’API
✅ Affichage label + confiance

🛑 Stopper tous les services
docker compose down -v
