import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

LABEL = "ASSALAMUALAIKUM" # Tukar label ni ikut isyarat
FILE_NAME = "data_bim.csv"
cap = cv2.VideoCapture(0)

print(f"Sedia rakam: {LABEL}. Tekan 's' simpan, 'q' keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1) & 0xFF
    data_row = [0] * 84 

    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i < 2: 
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                x_list = [lm.x for lm in hand_lms.landmark]
                y_list = [lm.y for lm in hand_lms.landmark]
                for j in range(21):
                    data_row[i*42 + j*2] = x_list[j] - min(x_list)
                    data_row[i*42 + j*2 + 1] = y_list[j] - min(y_list)

        if key == ord('s'):
            df = pd.DataFrame([data_row + [LABEL]])
            df.to_csv(FILE_NAME, index=False, mode='a', header=not os.path.exists(FILE_NAME))
            print(f"Data {LABEL} berjaya disimpan dalam {FILE_NAME}!")

    cv2.imshow("Rakam BIM", frame)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()