# 影片放置規則

## raw_sequences

剛錄好的原始影片先放：

- `data/videos/raw_sequences/`

## approved_sequences

審片通過後，移到：

- `data/videos/approved_sequences/`

## rerecord_needed

若審片發現影片不合格，需要補錄，放到：

- `data/videos/rerecord_needed/`

## admin15 行政句補錄

本次行政句任務新增在：

- `data/videos/raw_sequences/admin15/`
- `data/videos/approved_sequences/admin15/`
- `data/videos/rerecord_needed/admin15/`

三位組員資料夾代號：

- `LMX`：梁曼璇
- `XZQ`：徐姿琪
- `XCH`：謝承翰

每位組員都要完成：

- 15 句完整句子，各錄 `5` 遍
- 15 句句內詞，各錄 `5` 遍

資料夾結構：

```text
data/videos/<stage>/admin15/<member_code>_<name>/sentences/ADM15_Sxx_<sentence>/
data/videos/<stage>/admin15/<member_code>_<name>/words/ADM15_Sxx_<sentence>/Wxx_<word>/
```

檔名規則：

```text
完整句：ADM15_S01_LMX_01.mp4
單詞：ADM15_S01_W01_LMX_01.mp4
```

## 注意

- 標註組只標 `approved_sequences` 裡的影片
- `admin15` 的詞順沿用前次整理好的台灣手語詞序
- 每支影片請保留 `0.5 ~ 1` 秒起迄空白
- 句子與單詞影片都請錄滿 `5` 遍
