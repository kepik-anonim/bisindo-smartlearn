import streamlit as st
import base64
import os
import cv2
import numpy as np
import time
from PIL import Image
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import streamlit.components.v1 as components
import random
import gdown
import zipfile  # Tambahan untuk ekstrak ZIP

# ======== Tambahan untuk download model dari Google Drive ========
file_id = "17lqJ51NLGALZ2SZ6DB7RudVah8etIHtp"  # ID file di Google Drive
url = f"https://drive.google.com/uc?id={file_id}"
output = "model_bisindo.pkl"

if not os.path.exists(output):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(url, output, quiet=False)
# ================================================================

# ======== Ekstrak folder Contoh dari ZIP jika belum ada ========
quiz_zip = "Contoh.zip"
quiz_dir = "Contoh"
if not os.path.exists(quiz_dir):
    if os.path.exists(quiz_zip):
        with zipfile.ZipFile(quiz_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        st.warning("File Contoh.zip tidak ditemukan. Quiz tidak bisa dijalankan jika folder 'Contoh' tidak ada.")
# ================================================================

# Load model and preprocessing tools
model = joblib.load("model_bisindo.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Session states
if 'page' not in st.session_state:
    st.session_state.page = 'menu'
if 'show_nav' not in st.session_state:
    st.session_state.show_nav = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'quiz_index' not in st.session_state:
    st.session_state.quiz_index = 0
if 'quiz_files' not in st.session_state:
    if os.path.exists(quiz_dir):
        st.session_state.quiz_files = random.sample(
            [f for f in os.listdir(quiz_dir) if f.endswith((".jpg", ".png"))],
            len([f for f in os.listdir(quiz_dir) if f.endswith((".jpg", ".png"))])
        )
    else:
        st.session_state.quiz_files = []
if 'quiz_options' not in st.session_state:
    st.session_state.quiz_options = {}

# Background setter
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
            transition: background-image 0.8s ease-in-out;
        }}
    </style>
    """, unsafe_allow_html=True)

# Prediction function
def predict_letter(image):
    image = image.resize((100, 100)).convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0
    image_flat = image_array.flatten().reshape(1, -1)
    image_scaled = scaler.transform(image_flat)
    pred = model.predict(image_scaled)
    return label_encoder.inverse_transform(pred)[0]

# Video Transformer with black outside box
class SnapshotTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()
        mask = np.zeros_like(img)
        mask[50:300, 150:400] = img[50:300, 150:400]
        cv2.rectangle(mask, (150, 50), (400, 300), (255, 255, 0), 2)
        cv2.putText(mask, "Place your hands here ✋", (150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return mask

# Navigation Pane
def navigation_pane():
    if st.session_state.show_nav:
        with st.sidebar:
            if st.button("🏠 Home", key="nav_home"):
                st.session_state.page = 'menu'
                st.experimental_rerun()
            if st.button("📷 Learn with Camera", key="nav_cam"):
                st.session_state.page = 'camera'
                st.experimental_rerun()
            if st.button("🧩 Quiz", key="nav_quiz"):
                st.session_state.page = 'quiz'
                st.experimental_rerun()
            if st.button("🔤 Examples", key="nav_examples"):
                st.session_state.page = 'examples'
                st.experimental_rerun()

# Floating menu button
if st.button("☰", key="menu_btn_click"):
    st.session_state.show_nav = not st.session_state.show_nav
navigation_pane()

# Pages
if st.session_state.page == 'menu':
    set_background("background utama.png")
    st.markdown("<h1 style='text-align:center;color:#ff69b4;font-size:48px;font-weight:bold;'>🎈 BISINDO SmartLearn</h1>", unsafe_allow_html=True)
    st.markdown("""<div style='background-color:rgba(255,255,255,0.85);padding:15px;border-radius:10px;'>
    <p style='text-align:center;font-size:22px;font-weight:bold;'>BISINDO SmartLearn is an interactive and colorful platform for kids to learn Indonesian Sign Language. It aims to make learning fun, accessible, and engaging by combining real-time practice, quizzes, and visual examples.</p>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📷 Learn with Camera", use_container_width=True):
            st.session_state.page = 'camera'
            st.experimental_rerun()
    with col2:
        if st.button("🧩 Quiz", use_container_width=True):
            st.session_state.page = 'quiz'
            st.experimental_rerun()
    with col3:
        if st.button("🔤 A-Z Examples", use_container_width=True):
            st.session_state.page = 'examples'
            st.experimental_rerun()

elif st.session_state.page == 'camera':
    set_background("background menu.png")
    st.markdown("### 🖐️ Practice Sign Language - Snapshot Mode")
    col_cam, col_btn = st.columns([3, 1])
    with col_cam:
        ctx = webrtc_streamer(
            key="cam_stream", 
            video_transformer_factory=SnapshotTransformer,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_transform=True, 
            mode=WebRtcMode.SENDRECV
        )
    with col_btn:
        if ctx.video_transformer:
            if st.button("📸 Capture with Timer"):
                for sec in range(3, 0, -1):
                    st.write(f"Capturing in {sec}...")
                    time.sleep(1)
                frame = ctx.video_transformer.frame
                if frame is not None:
                    img_pil = Image.fromarray(cv2.cvtColor(frame[50:300, 150:400], cv2.COLOR_BGR2RGB))
                    st.session_state.captured_image = img_pil
                    pred = predict_letter(img_pil)
                    st.success(f"Detected Letter: {pred}")
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Captured Hand")

elif st.session_state.page == 'quiz':
    set_background("background menu.png")
    st.markdown("### 🧩 Guess the Sign Letter")
    if not st.session_state.quiz_files:
        st.error("Folder 'Contoh' tidak ditemukan atau kosong.")
    else:
        if st.session_state.quiz_index >= len(st.session_state.quiz_files):
            st.success(f"Quiz complete! Your score: {st.session_state.score}/{len(st.session_state.quiz_files)}")
        else:
            file = st.session_state.quiz_files[st.session_state.quiz_index]
            true_label = os.path.splitext(file)[0].upper()
            st.image(os.path.join(quiz_dir, file), width=200)

            if st.session_state.quiz_index not in st.session_state.quiz_options:
                opts = random.sample([chr(i) for i in range(65, 91) if chr(i) != true_label], 3) + [true_label]
                random.shuffle(opts)
                st.session_state.quiz_options[st.session_state.quiz_index] = opts
            else:
                opts = st.session_state.quiz_options[st.session_state.quiz_index]

            ans = st.radio("Choose the letter:", opts, key=f"quiz_radio_{st.session_state.quiz_index}")

            if st.button("Submit"):
                if ans == true_label:
                    st.success("✅ Correct!")
                    st.balloons()
                    components.html('<audio autoplay><source src="https://www.soundjay.com/human/kids-yeah.mp3" type="audio/mpeg"></audio>', height=0)
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Wrong! The correct answer is **{true_label}**")
                    components.html('<audio autoplay><source src="https://www.soundjay.com/human/kids-oh.mp3" type="audio/mpeg"></audio>', height=0)

            if st.button("Next Question"):
                st.session_state.quiz_index += 1
                st.experimental_rerun()

elif st.session_state.page == 'examples':
    set_background("background menu.png")
    st.markdown("### 🔤 A-Z Sign Language Examples")
    if os.path.exists(quiz_dir):
        files = sorted([f for f in os.listdir(quiz_dir) if f.endswith((".jpg", ".png"))])
        for file in files:
            st.image(os.path.join(quiz_dir, file), width=300, caption=f"Letter: {os.path.splitext(file)[0].upper()}")
    else:
        st.error("Folder 'Contoh' tidak ditemukan.")
