import cv2
import mediapipe as mp

print("--- SISTEM BERMULA ---")

# 1. Setup Kamera
cap = cv2.VideoCapture(0)

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

print("AI MediaPipe SEDIA! Angkat tangan awak...")

while True:
    success, img = cap.read()
    if not success:
        break

    # Tukar warna untuk AI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # BAHAGIAN LUKIS TITIK
    if results.multi_hand_landmarks:
        print("TANGAN DIKESAN!")
        for hand_lms in results.multi_hand_landmarks:
            # Melukis 21 titik (landmarks) dan garisan (connections)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    # Papar video
    cv2.imshow("TEST FYP ASSHAFIRAH", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()