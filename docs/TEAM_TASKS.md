# 團隊分工

## 1. 錄影組

負責人：
- 林芯羽
- 廖嘉儀

工作內容：
- 依照 `12` 組 sequence template 錄製影片
- 依命名規則輸出檔名
- 將影片放入 `data/videos/raw_sequences/`

會用到的檔案：
- [data/annotations/sequence_templates_12.csv](../data/annotations/sequence_templates_12.csv)
- [data/annotations/sequence_recording_manifest_300.csv](../data/annotations/sequence_recording_manifest_300.csv)

## 2. 審片 + 檔名整理組

負責人：
- 謝承翰

工作內容：
- 檢查影片是否需要重錄
- 確認 sequence 順序正確
- 確認畫面、光線、手部是否出框
- 整理合格影片與需重錄影片

輸出：
- 合格影片移到 `data/videos/approved_sequences/`
- 需重錄影片移到 `data/videos/rerecord_needed/`
- 填寫 review checklist

會用到的檔案：
- [data/annotations/sequence_recording_manifest_300.csv](../data/annotations/sequence_recording_manifest_300.csv)
- [data/annotations/sequence_review_checklist_300.csv](../data/annotations/sequence_review_checklist_300.csv)

## 3. 標註組

負責人：
- 梁曼璇
- 林姵妏

工作內容：
- 為每支 approved sequence 填寫 gloss 順序
- 補 `start_sec` / `end_sec`
- 需要時補 `notes`

會用到的檔案：
- [data/annotations/sequence_annotations_300_template.csv](../data/annotations/sequence_annotations_300_template.csv)
- [data/annotations/glossary_template.csv](../data/annotations/glossary_template.csv)

## 4. Gloss Description + 文件整理組

負責人：
- 徐姿琪

工作內容：
- 維護 glossary
- 為每個 gloss 撰寫 description
- 整理之後給 HST-SLR 使用的 gloss 描述資料

會用到的檔案：
- [data/annotations/glossary_template.csv](../data/annotations/glossary_template.csv)
- [data/annotations/gloss_description_assignment.csv](../data/annotations/gloss_description_assignment.csv)

## 5. GRU baseline 的定位

- `GRU baseline` = 目前已能跑通的基線
- `HST-SLR` = 下一階段高階連續辨識方向

所以目前所有 sequence 協作資料，除了支撐後續 HST-SLR，也會作為之後和 GRU baseline 比較的基礎。
