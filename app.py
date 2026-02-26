import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import librosa
import tempfile
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Multimodal AI System", layout="wide")
st.title("🚨 Multimodal Emergency Detection System")
st.markdown("### Vision + Audio + Text Integrated AI")

# -------------------------------
# Load Vision Model (Pretrained ResNet)
# -------------------------------
@st.cache_resource
def load_vision_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

vision_model = load_vision_model()

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Vision Prediction
# -------------------------------
def predict_image(image):
    img = vision_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vision_model(img)
        prob = torch.softmax(outputs, dim=1)
        score = float(torch.max(prob))
    return score


# -------------------------------
# Audio Prediction
# -------------------------------
def predict_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, duration=5)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)

    score = np.mean(mel_db)
    score = abs(score) / 100
    score = min(score, 1.0)

    return float(score)


# -------------------------------
# Text Model (Simple TF-IDF)
# -------------------------------
@st.cache_resource
def load_text_model():
    sample_texts = [
        "There is fire",
        "Help me please",
        "Everything is normal",
        "Smoke everywhere",
        "I am safe"
    ]
    labels = [1, 1, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sample_texts)
    clf = LogisticRegression()
    clf.fit(X, labels)

    return vectorizer, clf

vectorizer, text_model = load_text_model()

def predict_text(text):
    X = vectorizer.transform([text])
    prob = text_model.predict_proba(X)[0][1]
    return float(prob)


# -------------------------------
# Fusion Logic
# -------------------------------
def fuse_predictions(vision_score, audio_score, text_score):

    final_score = (
        0.4 * vision_score +
        0.3 * audio_score +
        0.3 * text_score
    )

    if final_score > 0.75:
        level = "HIGH 🔴"
    elif final_score > 0.4:
        level = "MEDIUM 🟠"
    else:
        level = "LOW 🟢"

    return final_score, level


# -------------------------------
# UI Layout
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

with col2:
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

with col3:
    text_input = st.text_area("Enter Text Description")


# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze Situation"):

    vision_score = 0
    audio_score = 0
    text_score = 0

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        vision_score = predict_image(image)

    if audio_file:
        audio_score = predict_audio(audio_file)

    if text_input:
        text_score = predict_text(text_input)

    final_score, level = fuse_predictions(vision_score, audio_score, text_score)

    # Build a human-readable situation summary using simple heuristics
    situation_parts = []
    text_lower = (text_input or "").lower()
    if "fire" in text_lower or text_score > 0.6:
        situation_parts.append("Fire Emergency")
    if any(k in text_lower for k in ("down", "fallen", "person down", "injured")) or vision_score > 0.65:
        situation_parts.append("Person Down")
    if audio_score > 0.6 or any(k in text_lower for k in ("scream", "screaming", "help")):
        if "Screaming / Panic" not in situation_parts:
            situation_parts.append("Screaming / Panic")
    situation = " + ".join(situation_parts) if situation_parts else "Unspecified"

    # Format the report block to match the requested layout
    confidence_pct = round(final_score * 100, 1)
    report_lines = [
        "MULTIMODAL EMERGENCY RISK DETECTION SYSTEM",
        "============================================================",
        "",
        f" Risk Level: {level}",
        f" Confidence: {confidence_pct}%",
        f" Situation: {situation}",
        "",
        " Recommended Actions:",
        "   ⚠️  Prepare to evacuate",
        "   📱 Keep phone accessible",
        "   🚪 Note exits and safe locations",
        "   👥 Alert nearby individuals",
        "   ⏱️  Monitor situation closely",
        "",
        " Modality Breakdown:",
        f"   Vision Score: {vision_score:.3f}",
        f"   Audio Score: {audio_score:.3f}",
        f"   Text Score: {text_score:.3f}",
        "",
        "============================================================",
    ]

    st.markdown("\n".join(["### " + report_lines[0]]))
    st.code("\n".join(report_lines[1:]), language=None)

    # Also show a compact metric and progress bar for quick glance
    st.markdown("### Quick Summary")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric("Risk", level, delta=f"{confidence_pct}%")
    with col_b:
        st.progress(min(max(final_score, 0.0), 1.0))
