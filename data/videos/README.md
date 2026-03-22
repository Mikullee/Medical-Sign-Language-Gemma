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

## 注意

- 標註組只標 `approved_sequences` 裡的影片
- 原始影片命名必須符合：

```text
SEQ編號_錄影者代號_次數.mp4
```
