# 組員任務說明

## 1. 錄影組

負責：

- 依照 `12` 組 sequence template 錄影片
- 檔名要完全照規則
- 錄好的影片先放到：
  - `data/videos/raw_sequences/`

要看的檔案：

- `data/annotations/sequence_templates_12.csv`
- `data/annotations/sequence_recording_manifest_300.csv`

要更新的檔案：

- `data/annotations/sequence_recording_manifest_300.csv`

---

## 2. 審片 + 檔名整理組

負責：

- 檢查影片有沒有錄好
- 檢查手是否出框、光線是否穩定、順序是否正確
- 判斷需不需要重錄
- 整理檔名與資料夾

影片整理規則：

- 合格影片放到：
  - `data/videos/approved_sequences/`
- 不合格、待補錄影片放到：
  - `data/videos/rerecord_needed/`

要看的檔案：

- `data/annotations/sequence_recording_manifest_300.csv`
- `data/annotations/sequence_review_checklist_300.csv`

要更新的檔案：

- `data/annotations/sequence_review_checklist_300.csv`

---

## 3. 標註組

負責：

- 針對審片通過的影片做 gloss 標註
- 填每個 gloss 的開始與結束時間

標註規則：

- 一列只標一個 gloss
- 用秒數填 `start_sec` / `end_sec`
- 建議精度到小數三位
- 如果不確定，先寫在 `notes`

只標註這個資料夾內的影片：

- `data/videos/approved_sequences/`

要看的檔案：

- `data/annotations/sequence_annotations_300_template.csv`
- `data/annotations/glossary_template.csv`

要更新的檔案：

- `data/annotations/sequence_annotations_300_template.csv`

---

## 4. Gloss Description + 文件整理組

負責：

- 維護 gloss 的標準名稱
- 整理 alias / 同義詞
- 撰寫每個 gloss 的動作描述

要看的檔案：

- `data/annotations/glossary_template.csv`
- `data/annotations/gloss_description_assignment.csv`

要更新的檔案：

- `data/annotations/glossary_template.csv`
- `data/annotations/gloss_description_assignment.csv`

注意：

- description 是「每個 gloss 一份」，不是每支影片一份
- `其他` 只能表示「其他地區」的其他
