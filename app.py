import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import mediapipe as mp

# Load model + labels
clf = joblib.load("bsl_landmark_model.joblib")
label_map = pd.read_csv("label_map.csv")
id_to_label = dict(zip(label_map["label_id"], label_map["label"]))

# MediaPipe init (reuse once)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks_rgb(rgb):
    """
    rgb: numpy array shape (H,W,3), dtype uint8, RGB order
    returns: (126,) float32
    """
    results = hands_detector.process(rgb)

    if not results.multi_hand_landmarks:
        return np.zeros((126,), dtype=np.float32)

    feats = []
    for hand_idx in range(2):
        if hand_idx < len(results.multi_hand_landmarks):
            hand = results.multi_hand_landmarks[hand_idx]
            for lm in hand.landmark:
                feats.extend([lm.x, lm.y, lm.z])
        else:
            feats.extend([0.0] * (21 * 3))

    return np.array(feats, dtype=np.float32)

st.title("BSL Hand Gesture Classifier (Landmarks)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    rgb = np.array(img)  # (H,W,3) uint8 RGB

    st.image(img, caption="Uploaded image", use_container_width=True)

    X_one = extract_landmarks_rgb(rgb).reshape(1, -1)
    pred_id = int(clf.predict(X_one)[0])
    pred_label = id_to_label.get(pred_id, str(pred_id))

    st.success(f"Prediction: {pred_label} (id={pred_id})")
