import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 設定與路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
actions = np.array(['ache', 'fever']) # 你的動作標籤
no_sequences = 30
sequence_length = 30

# 2. 建立標籤映射 (ache->0, fever->1)
label_map = {label:num for num, label in enumerate(actions)}

# 3. 載入數據
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, f"{sequence}_{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences) # 形狀: (60, 30, 63)
y = to_categorical(labels).astype(int)

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 4. 建立 LSTM 模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 5. 開始訓練
print("開始訓練 AI 模型...")
model.fit(X_train, y_train, epochs=200) # 訓練 200 次

# 6. 儲存模型
model.save('medical_sign_model.h5')
print("✅ 模型已訓練完成並儲存為 medical_sign_model.h5")