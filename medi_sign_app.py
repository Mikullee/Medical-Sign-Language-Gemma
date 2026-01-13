import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. 模型結構定義 (維持 30 幀) ---
def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63))) 
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

# --- 2. 載入權重 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "medical_sign_model.h5")

model = build_model()
try:
    model.load_weights(model_path)
    print("✅ 醫無礙大腦已成功啟動！")
except Exception as e:
    print(f"❌ 載入失敗：{e}")
    exit()

# --- 3. 初始化 Mediapipe ---
actions = ['ache', 'fever'] 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- 4. 變數設定 ---
sequence = []
threshold = 0.5 
last_text = "" 
display_frames = 0 

cap = cv2.VideoCapture(0)



while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # --- 修改邏輯：判斷畫面上是否有手 ---
    if results.multi_hand_landmarks:
        # 有手時：繪製節點並累積數據
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 提取座標
        keypoints = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        sequence.append(keypoints)
        sequence = sequence[-30:] 

        # 進行預測
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            if res[np.argmax(res)] > threshold:
                action_name = actions[np.argmax(res)]
                last_text = "PAIN" if action_name == 'ache' else "FEVER"
                display_frames = 30 # 在有手的情況下維持顯示
    else:
        # --- 沒手時：立即清空所有狀態 ---
        sequence = []      # 清空序列，避免數據斷層導致誤判
        display_frames = 0 # 立即隱藏藍色提示框

    # --- 繪製邏輯：只有「有手」且「有結果」時才畫框 ---
    if display_frames > 0 and results.multi_hand_landmarks:
        cv2.rectangle(image, (0,0), (640, 55), (245, 117, 16), -1)
        cv2.putText(image, f'DIAGNOSIS: {last_text}', (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        display_frames -= 1 

    cv2.imshow('Medi-Sign Connect - Realtime', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()