import cv2
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import pygame
import os

# --- 1. SETUP SUARA (GOOGLE) ---
pygame.mixer.init()
def sebut_google(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "suara.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): continue
        pygame.mixer.music.unload()
    except: pass

# --- 2. TRAINING AI ---
print("--- SISTEM SEDANG BELAJAR ---")
data = pd.read_csv('data_asl.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# --- 3. SETUP VIDEO ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

last_prediction = ""
counter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # --- DESIGN: HEADER BAR ---
    # Letak bar hitam kat atas untuk nampak premium
    cv2.rectangle(img, (0,0), (w, 50), (30, 30, 30), -1)
    cv2.putText(img, "ASSHAFIRAH'S ASL TRANSLATOR V1.0", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            koordinat_live = []
            for lm in hand_lms.landmark:
                koordinat_live.append(lm.x)
                koordinat_live.append(lm.y)
            
            prediction = model.predict([koordinat_live])[0]
            
            if prediction == last_prediction:
                counter += 1
            else:
                counter = 0
                last_prediction = prediction

            if counter == 8: # Lagi cepat sikit (8 frame)
                sebut_google(prediction)
                counter = 9

            # Lukis titik jari dengan warna tema (Ungu/Biru)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS,
                                 mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                                 mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2))
            
            # --- DESIGN: FLOATING BOX ---
            # Kotak keputusan yang nampak "modern"
            cv2.rectangle(img, (20, h-100), (220, h-20), (255, 0, 255), 2) # Border ungu
            cv2.rectangle(img, (20, h-100), (220, h-20), (50, 50, 50), -1) # Isi kelabu gelap
            cv2.putText(img, "DETECTED:", (35, h-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, str(prediction), (35, h-35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        last_prediction = ""
        counter = 0
        cv2.putText(img, "WAITING FOR HAND...", (w//2-100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("PROJEK FYP ASSHAFIRAH", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()