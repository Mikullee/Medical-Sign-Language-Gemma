# Medical Sign Language Project

## 這個 repo 是做什麼的

這個 repo 目前是給組員協作使用的工作區，主要用途是：

- 錄 `300` 支 sequence 影片
- 審片與整理檔名
- 標註 gloss 順序與 `start/end time`
- 撰寫 glossary 與 gloss description

目前辨識端 baseline 方向是：

- `MediaPipe`
- `pose + hand landmarks`
- 相對座標 / 幾何特徵 / 速度特徵
- `Bidirectional GRU`

後續目標是：

- 把 sequence 資料擴大
- 再往 `HST-SLR` 方向整理資料與訓練

---

## 組員先看這三個檔案

1. [`docs/TEAM_TASKS.md`](docs/TEAM_TASKS.md)
2. [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md)
3. [`data/annotations/sequence_work_plan_300.md`](data/annotations/sequence_work_plan_300.md)

---

## 資料夾怎麼用

### 影片放這裡

- 原始剛錄好的影片：
  - [`data/videos/raw_sequences/`](data/videos/raw_sequences/)
- 審片通過後的影片：
  - [`data/videos/approved_sequences/`](data/videos/approved_sequences/)
- 需要重錄的影片：
  - [`data/videos/rerecord_needed/`](data/videos/rerecord_needed/)

### 標註與任務表放這裡

- [`data/annotations/`](data/annotations/)

這裡面最重要的檔案有：

- [`sequence_templates_12.csv`](data/annotations/sequence_templates_12.csv)
- [`sequence_recording_manifest_300.csv`](data/annotations/sequence_recording_manifest_300.csv)
- [`sequence_review_checklist_300.csv`](data/annotations/sequence_review_checklist_300.csv)
- [`sequence_annotations_300_template.csv`](data/annotations/sequence_annotations_300_template.csv)
- [`glossary_template.csv`](data/annotations/glossary_template.csv)
- [`gloss_description_assignment.csv`](data/annotations/gloss_description_assignment.csv)

---

## 每組要做什麼

### 錄影組

- 依照 `12` 組 template 錄完 `300` 支影片
- 檔名一定要照規則
- 影片先放到：
  - [`data/videos/raw_sequences/`](data/videos/raw_sequences/)

### 審片 + 檔名整理組

- 檢查影片合不合格
- 需要重錄的移到：
  - [`data/videos/rerecord_needed/`](data/videos/rerecord_needed/)
- 合格的移到：
  - [`data/videos/approved_sequences/`](data/videos/approved_sequences/)
- 同時更新：
  - [`sequence_review_checklist_300.csv`](data/annotations/sequence_review_checklist_300.csv)

### 標註組

- 只標註審片通過的影片
- 填：
  - [`sequence_annotations_300_template.csv`](data/annotations/sequence_annotations_300_template.csv)
- 每個 gloss 要填：
  - `start_sec`
  - `end_sec`

### Gloss Description + 文件整理組

- 維護 glossary
- 寫每個 gloss 的 description
- 更新：
  - [`glossary_template.csv`](data/annotations/glossary_template.csv)
  - [`gloss_description_assignment.csv`](data/annotations/gloss_description_assignment.csv)

---

## 命名規則

影片命名格式：

```text
SEQ編號_錄影者代號_次數.mp4
```

範例：

```text
SEQ01_A_01.mp4
SEQ01_C_01.mp4
SEQ12_A_12.mp4
```

---

## 錄影注意事項

- 上半身完整入鏡
- 手不要出框
- 鏡頭固定
- 背景盡量單純
- 光線穩定
- 每支影片開頭和結尾各留 `0.5 ~ 1` 秒空白
- gloss 之間銜接自然，不要太快

特別注意：

- `其他` 只限定成「其他地區」這個語境
- 不可把不確定或模糊段落都標成 `其他`

---

## 目前這輪的目標

- `12` 組 template
- `300` 支 sequence 影片
- 補完整的 gloss 順序標註
- 至少補一批 `start/end` 時間標註
- 整理 glossary 與 gloss description

這一輪完成後，資料才比較有條件往 `HST-SLR` 方向推進。
