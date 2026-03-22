# 檔案用途說明

## `data/annotations/sequence_templates_12.csv`

用途：

- 列出這次的 `12` 組 sequence 模板
- 給錄影組、審片組、標註組確認 gloss 順序

---

## `data/annotations/sequence_recording_manifest_300.csv`

用途：

- 這次 `300` 支影片的總控表
- 誰錄哪一支、哪一組 template、錄第幾次，都看這份

---

## `data/annotations/sequence_review_checklist_300.csv`

用途：

- 審片用檢查表
- 記錄影片是否合格、是否需要重錄

---

## `data/annotations/sequence_annotations_300_template.csv`

用途：

- 標註組填 gloss 的 `start_sec / end_sec`
- 一列對應一個 gloss 區段

---

## `data/annotations/glossary_template.csv`

用途：

- glossary 主表
- 統一 gloss 名稱、意思、alias、description

---

## `data/annotations/gloss_description_assignment.csv`

用途：

- gloss description 的工作表
- 記錄哪些 gloss 已寫、哪些還沒寫

---

## `data/annotations/sequence_work_plan_300.md`

用途：

- 這輪任務的完整說明
- 包含分工、錄影規則、template、工作內容

---

## `data/videos/raw_sequences/`

用途：

- 錄影組剛錄好的原始影片先放這裡

---

## `data/videos/approved_sequences/`

用途：

- 審片通過後的正式影片放這裡
- 標註組只處理這裡的影片

---

## `data/videos/rerecord_needed/`

用途：

- 審片後判定要重錄的影片放這裡
