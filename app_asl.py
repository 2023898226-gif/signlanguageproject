import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading

# 1. Setup Suara (Versi Re-initialization)
def speak(text):
    def run_speech():
        try:
            # Kita init engine kat dalam ni supaya dia tak 'hang'
            engine = pyttsx3.init()
            # Tukar speed (optional, 150 nampak lebih natural)
            engine.setProperty('rate', 150) 
            engine.say(text)
            engine.runAndWait()
            engine.stop() # Tutup balik lepas cakap
        except:
            pass

    threading.Thread(target=run_speech, daemon=True).start()

# 2. Setup MediaPipe & Model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    st.error(f"Gagal load model: {e}")

st.title("🤟 ASL Voice-Activated AI")
run = st.checkbox('Aktifkan Sistem')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

last_spoken = "" 

while run:
    ret, frame = cam.read()
    if not ret: 
        st.write("Kamera gagal dikesan.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            if len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    char = prediction[0]
                    
                    # Lukis Bayang (Shadow) - Kasi cantik!
                    cv2.putText(frame, f"Detected: {char}", (32, 52), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
                    # Tulisan Utama
                    cv2.putText(frame, f"Detected: {char}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 127), 2, cv2.LINE_AA)
                    
                    # Logic Suara: Panggil sekali saja
                    if char != last_spoken:
                        speak(char)
                        last_spoken = char
                        
                except Exception as e:
                    print(f"Predict Error: {e}")

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    cam.release()
    cv2.destroyAllWindows()