import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- KONFIGURASI ---
label_nama = "B" 
data_senarai = []
FILE_NAME = "data_asl.csv"

cap = cv2.VideoCapture(0)

print(f"Sedia untuk kumpul data huruf: {label_nama}")
print("Tekan 's' untuk simpan data. Tekan 'q' untuk berhenti.")

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Kita baca keyboard SEKALI SAJA di sini untuk elak lag
    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Jika tekan 's'
            if key == ord('s'):
                x_ = [lm.x for lm in hand_lms.landmark]
                y_ = [lm.y for lm in hand_lms.landmark]
                
                koordinat_relatif = []
                for i in range(len(hand_lms.landmark)):
                    koordinat_relatif.append(x_[i] - min(x_))
                    koordinat_relatif.append(y_[i] - min(y_))
                
                koordinat_relatif.append(label_nama)
                data_senarai.append(koordinat_relatif)
                print(f"Data {label_nama} disimpan! Jumlah: {len(data_senarai)}")

    cv2.putText(img, f"Huruf: {label_nama} | Count: {len(data_senarai)}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Kumpul Data ASL (Manual)", img)
    
    # Keluar jika tekan 'q'
    if key == ord('q'):
        break

# 3. Simpan ke CSV
if len(data_senarai) > 0:
    df = pd.DataFrame(data_senarai)
    # Gunakan header=False kalau fail dah wujud supaya column name tak berulang kat tengah-tengah
    fail_wujud = os.path.exists(FILE_NAME)
    df.to_csv(FILE_NAME, index=False, mode='a', header=not fail_wujud)
    print(f"Siap! {len(data_senarai)} data baru masuk dalam {FILE_NAME}")
else:
    print("Eh, awak tak tekan 's' tadi ke? Tiada data disimpan.")

cap.release()
cv2.destroyAllWindows()