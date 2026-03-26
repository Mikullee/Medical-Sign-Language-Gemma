## Branch note

The uploaded raw sequence videos are currently maintained in branch `dev-agent-sync`.

If this folder looks empty in `main`, view the branch below:

- [View `dev-agent-sync` branch](https://github.com/Mikullee/Medical-Sign-Language-Gemma/tree/dev-agent-sync)

中文說明：

- `main` 這裡可能只保留資料夾結構。
- 目前已上傳的 raw 影片在 `dev-agent-sync`。

# 影片資料夾使用方式

## raw_sequences

位置：[raw_sequences](./raw_sequences)

用途：
- 錄影組剛錄完、尚未審核的 sequence 影片先放這裡

## approved_sequences

位置：[approved_sequences](./approved_sequences)

用途：
- 審片後確認可用的影片放這裡
- 標註組只處理這個資料夾內的影片

## rerecord_needed

位置：[rerecord_needed](./rerecord_needed)

用途：
- 有問題、需要重錄的影片放這裡

## 錄影命名格式

```text
SEQ編號_錄影者代號_次數.mp4
```

範例：

```text
SEQ01_A_01.mp4
SEQ01_C_01.mp4
SEQ12_A_12.mp4
```

## 錄影規則

- 上半身完整入鏡
- 手不要出框
- 鏡頭固定
- 背景盡量單純
- 光線穩定
- 每支影片開頭和結尾各留 `0.5 ~ 1` 秒空白
- gloss 之間銜接自然，不要太快
- `其他` 只限定在「其他地區」語境
