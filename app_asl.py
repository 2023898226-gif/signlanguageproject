import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Pastikan path model betul. Kalau model.p ada dalam folder yang sama:
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except Exception as e:
    st.error(f"Model tak jumpa: {e}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                char = str(prediction[0])
                cv2.putText(img, f"Detected: {char}", (30, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 127), 2)

    return frame.from_ndarray(img, format="bgr24")

st.title("🤟 ASL Real-Time Translator")

webrtc_streamer(
    key="asl-translator",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
