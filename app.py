import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import mediapipe as mp

# Load model + label map
clf = joblib.load("bsl_landmark_model.joblib")
label_map = pd.read_csv("label_map.csv")
id_to_label = dict(zip(label_map["label_id"], label_map["label"]))

# Setup MediaPipe once
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks_fast(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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

# ---- UI ----
st.title("BSL Hand Gesture Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    feats = extract_landmarks_fast(bgr).reshape(1, -1)
    pred_id = int(clf.predict(feats)[0])
    pred_label = id_to_label[pred_id]

    st.success(f"Prediction: {pred_label}")
