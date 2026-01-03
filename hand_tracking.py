import cv2
import mediapipe as mp

# 1. 初始化 Mediapipe 的手部偵測模組
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # 醫療手語有時會用到雙手
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

# 2. 開啟電腦視訊鏡頭
cap = cv2.VideoCapture(0)

print("程式啟動中... 請對著鏡頭比出手勢。按 'q' 鍵可退出。")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("無法取得畫面")
        break

    # 為了辨識，需先將 BGR 轉為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 3. 畫出手部節點
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 畫出點跟線
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # 顯示畫面
    cv2.imshow('Medical Sign Language - Base', image)

    # 按 'q' 鍵退出程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()