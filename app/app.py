import os
import io
import requests
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Emotion Classifier", page_icon="🙂", layout="centered")

st.title("🎭 Facial Emotion Classifier")
st.caption("Upload une image, et l'API prédit l'émotion.")

# ✅ Récupère par défaut l’URL fournie par Docker Compose
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8080/predict")

api_url = st.text_input(
    "API URL",
    DEFAULT_API_URL,
    help="Endpoint POST /predict de ton API FastAPI"
)

tab1, tab2 = st.tabs(["📤 Upload", "🔗 Par URL"])

image_bytes = None

# ----------- UPLOAD LOCAL -----------
with tab1:
    up = st.file_uploader("Choisis une image (jpg/png)", type=["jpg","jpeg","png"])
    if up:
        image_bytes = up.getvalue()
        st.image(Image.open(io.BytesIO(image_bytes)), caption="Aperçu")

# ----------- CHARGEMENT VIA URL -----------
with tab2:
    url = st.text_input("URL d'image (jpg/png)")
    if st.button("Charger depuis l'URL"):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            image_bytes = r.content
            st.image(Image.open(io.BytesIO(image_bytes)), caption="Aperçu")
        except Exception as e:
            st.error(f"Impossible de télécharger l'image: {e}")

st.divider()

# ----------- PRÉDICTION -----------
if image_bytes and st.button("🔮 Prédire"):
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(api_url, files=files, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        st.success(f"✅ Emotion: **{data.get('label', '?')}** — confiance: **{data.get('confidence', 0):.3f}**")
        st.json(data)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur requête API: {e}")
    except Exception as e:
        st.error(f"Erreur: {e}")

st.caption("Astuce: change l'URL de l'API si nécessaire.")
