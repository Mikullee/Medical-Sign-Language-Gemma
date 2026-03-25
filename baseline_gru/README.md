# GRU Baseline

這個資料夾整理的是目前專案已完成、可展示、可比較的 `BiGRU` 基線版本。

## 1. 這版模型在做什麼

目前辨識流程是：

`MediaPipe -> pose + hand landmarks -> relative/geometry/velocity features -> Bidirectional GRU -> gloss prediction`

用途：
- 先驗證手語影片是否能輸出 gloss 預測
- 作為目前可運作的 baseline
- 提供之後和 `HST-SLR` 比較的基準

## 2. 用到哪些素材

這版主要用到三種資料來源：

- 舊的多支 gloss 資料：`curated_gloss`
- 新補錄的 gloss 資料：`GLOSS+`
- 已標註 sequence 與部分 `BLANK` 過渡段

主測試影片是：
- `Video Project.mp4`

## 3. 兩個版本差異

### Best numeric 版

- 以 gloss 訓練為主
- 驗證數值最好
- `validation accuracy = 0.7976`
- 對應權重：
  - `models/gloss_gru_model.pt`
- 對應示範影片：
  - `results/video_project_gru_comparison_clear.mp4`

### Mixed + sequence + blank 版

- 納入 sequence 與 `BLANK` 過渡段
- 更接近連續辨識方向
- `validation accuracy = 0.7546`
- 對應權重：
  - `models/gloss_gru_mixed_model.pt`
- 對應示範影片：
  - `results/video_project_gru_mixed_comparison_clear.mp4`

## 4. 目前最好的結果是哪一版

如果看純數值，目前表現較好的是 **best numeric 版**。

如果看研究方向，**mixed + sequence + blank 版** 更接近後續要做的連續手語辨識。

這裡的 `validation accuracy` 是：

**以固定時間窗 landmarks 片段為單位，計算 gloss 分類正確率。**

它不是整段 sequence 完全正確率，也不是整支影片翻譯正確率。

## 5. 為什麼後續還要往 HST-SLR 走

目前 `BiGRU` 雖然能跑通，但仍有兩個主要限制：

- 對 gloss 開始 / 結尾的 segmentation 還不夠準
- 對連續手語中的過渡動作處理有限

因此：
- `GRU baseline` = 目前可運作的基線
- `HST-SLR` = 下一階段高階連續辨識方向

## 可重跑腳本

目前保留的核心腳本在 [scripts](./scripts)：

- `train_gloss_gru.py`
- `train_gloss_gru_mixed.py`
- `realtime_infer_gru.py`
- `render_video_comparison_best_clear.py`
- `render_video_comparison_mixed.py`

## 目前保留的結果

### Models
- `models/gloss_gru_model.pt`
- `models/gloss_gru_mixed_model.pt`
- `models/hand_landmarker.task`
- `models/pose_landmarker.task`

### Metrics / Reports
- `results/gru_training_metrics.json`
- `results/gru_mixed_training_metrics.json`
- `results/video_project_gru_comparison.csv`
- `results/video_project_gru_mixed_comparison.csv`
- `results/video_project_gru_report.md`

### Demo videos
- `results/video_project_gru_comparison_clear.mp4`
- `results/video_project_gru_mixed_comparison_clear.mp4`

## 使用說明

### 直接查看成果

教授或組員如果只是要看結果，直接打開：

- `results/video_project_gru_comparison_clear.mp4`
- `results/video_project_gru_mixed_comparison_clear.mp4`

### 即時辨識

若要直接使用目前保留的 baseline 權重做即時辨識：

```powershell
python baseline_gru/scripts/realtime_infer_gru.py
```

### 重跑訓練或重新輸出 comparison

保留的訓練與 comparison 腳本已改成 repo-relative，但它們仍需要本機另外有原始資料。

若你的原始資料不是放在 repo 根目錄，而是放在其他工作資料夾，可以先設定：

```powershell
$env:MSG_DATA_ROOT="你的工作資料夾路徑"
```

之後再執行：

```powershell
python baseline_gru/scripts/train_gloss_gru.py
python baseline_gru/scripts/train_gloss_gru_mixed.py
python baseline_gru/scripts/render_video_comparison_best_clear.py
python baseline_gru/scripts/render_video_comparison_mixed.py
```
