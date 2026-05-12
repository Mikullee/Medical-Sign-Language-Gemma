# 手語辨識專案（dev-agent-sync）

## 目前狀態

1. 300 部 sequence 已完成錄製（raw）。
2. 目前進入補強階段：補低 token gloss、審片、時間標註。
3. 2026-05-12 新增 `admin15` 行政句錄影資料夾與說明，供三位組員同步補錄 15 句完整句子與句內詞。

## 新錄製清單（請先看）

請直接使用這兩份檔案：

1. `data/annotations/seq_boost_plan_project_order_gap70.csv`
2. `data/annotations/seq_boost_plan_project_order_gap70.md`

以上檔案已定義：
- 要錄哪些 sequence（P01/P02）
- 每個 sequence 要錄幾次
- 這批錄完會補到哪些 gloss token 缺口

## 行政 15 句補錄

這次新增的行政句任務請看：

1. `docs/ADMIN15_RECORDING_GUIDE.md`
2. `data/videos/README.md`

這批資料已經建立在：

1. 未審原始影片：`data/videos/raw_sequences/admin15/`
2. 審片通過影片：`data/videos/approved_sequences/admin15/`
3. 需要重錄影片：`data/videos/rerecord_needed/admin15/`

## 資料放置位置

1. 新拍未審影片：
- `data/videos/raw_sequences/<sequence_id>/`

2. 審片通過：
- `data/videos/approved_sequences/<sequence_id>/`

3. 需重錄：
- `data/videos/rerecord_needed/<sequence_id>/`

## 必更新標註檔

1. `data/annotations/sequence_recording_manifest_300.csv`
2. `data/annotations/sequence_review_checklist_300.csv`
3. `data/annotations/sequence_annotations_300_template.csv`

## 分支用途

1. `dev-agent-sync`：工作中資料（raw、同步中）
2. `main`：穩定可訓練資料（approved + annotation）
