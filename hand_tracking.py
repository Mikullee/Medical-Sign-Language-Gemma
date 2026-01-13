import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- 1. 設定區 ---
# Windows 系統強迫存到桌面
DATA_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "Medical_Sign_Data")  # 資料儲存根目錄
action = "fever"                # 正在錄製的動作名稱 (例如：發燒)
no_sequences = 30               # 預計錄製幾組動作
sequence_length = 30            # 每一組動作錄製幾幀 (影格)

# 建立資料夾
if not os.path.exists(os.path.join(DATA_PATH, action)):
    os.makedirs(os.path.join(DATA_PATH, action))

# --- 2. 初始化 Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) # 如果是 iVCam 請嘗試改為 1

# 輔助函式：提取座標並轉為 numpy 陣列
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        # 這裡簡單處理：只取第一隻偵測到的手
        lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
    else:
        # 如果沒偵測到，填充 0
        lh = np.zeros(21*3)
    return lh

# --- 3. 開始錄製流程 ---
for sequence in range(no_sequences):
    for frame_num in range(sequence_length):
        success, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # 繪製畫面
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 顯示錄製提示文字
        if frame_num == 0:
            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Medical Sign Language - Data Collection', image)
            cv2.waitKey(2000) # 每組動作間隔 2 秒，讓你準備
        else: 
            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Medical Sign Language - Data Collection', image)

        # 儲存關鍵點數據
        keypoints = extract_keypoints(results)
        npy_path = os.path.join(DATA_PATH, action, f"{sequence}_{frame_num}.npy")
        np.save(npy_path, keypoints)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()