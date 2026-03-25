# Repo 文件與資料夾用途

## 根目錄

- [README.md](../README.md)
  - 專案入口說明
- [baseline_gru](../baseline_gru)
  - 目前 GRU baseline 成果與可重跑腳本

## docs

- [TEAM_TASKS.md](./TEAM_TASKS.md)
  - 組員分工與每組輸出
- [FILE_GUIDE.md](./FILE_GUIDE.md)
  - 本文件，說明 repo 裡每個重要檔案是做什麼的

## data/annotations

- [sequence_work_plan_300.md](../data/annotations/sequence_work_plan_300.md)
  - 300 支 sequence 的完整任務規劃
- [sequence_templates_12.csv](../data/annotations/sequence_templates_12.csv)
  - 12 組 sequence template
- [sequence_recording_manifest_300.csv](../data/annotations/sequence_recording_manifest_300.csv)
  - 錄影總表，列出 300 支影片的檔名與分配
- [sequence_review_checklist_300.csv](../data/annotations/sequence_review_checklist_300.csv)
  - 審片檢查表
- [sequence_annotations_300_template.csv](../data/annotations/sequence_annotations_300_template.csv)
  - 標註組填 `start_sec / end_sec` 的模板
- [glossary_template.csv](../data/annotations/glossary_template.csv)
  - gloss 標準名稱、意思、別名、描述模板
- [gloss_description_assignment.csv](../data/annotations/gloss_description_assignment.csv)
  - gloss description 的分工與完成狀態

## data/videos

- [raw_sequences](../data/videos/raw_sequences)
  - 錄影組剛錄完、尚未審核的 sequence 影片
- [approved_sequences](../data/videos/approved_sequences)
  - 審片後確認可用的 sequence 影片
- [rerecord_needed](../data/videos/rerecord_needed)
  - 需要重錄的影片
- [README.md](../data/videos/README.md)
  - 影片資料夾使用規則

## baseline_gru

- [README.md](../baseline_gru/README.md)
  - GRU baseline 的技術與結果總說明
- `scripts/`
  - 可重跑的 GRU 訓練、推論、comparison 渲染腳本
- `models/`
  - 目前保留的兩個 GRU 權重
- `results/`
  - metrics、comparison CSV、結果摘要與兩支示範影片
