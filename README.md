# Medical Sign Language Gemma

這個 repo 目前有兩條主線：

1. `sequence` 資料蒐集與標註協作
2. `GRU baseline` 成果整理與展示

目前 `GRU baseline` 是可運作的基線；`HST-SLR` 是下一階段要評估的高階連續手語辨識方向。

## 先看哪裡

組員與老師如果第一次進 repo，先看這三個檔案：

1. [docs/TEAM_TASKS.md](./docs/TEAM_TASKS.md)
2. [docs/FILE_GUIDE.md](./docs/FILE_GUIDE.md)
3. [baseline_gru/README.md](./baseline_gru/README.md)

## 目前在做什麼

### 1. Sequence 協作資料

- 目標：建立 `300` 支 sequence 影片
- 內容：錄影、審片、標註 `gloss sequence` 與部分 `start/end time`
- 主要檔案放在：
  - [data/annotations](./data/annotations)
  - [data/videos/raw_sequences](./data/videos/raw_sequences)
  - [data/videos/approved_sequences](./data/videos/approved_sequences)
  - [data/videos/rerecord_needed](./data/videos/rerecord_needed)

### 2. GRU baseline

- 位置：[baseline_gru](./baseline_gru)
- 內容：
  - 目前可重跑的 `BiGRU` 訓練與推論腳本
  - 兩個版本的模型權重
  - 主影片 comparison CSV
  - 兩支示範影片
- 用途：
  - 給教授看目前成果
  - 當之後 `HST-SLR` 的比較基線

## 影片與標註放置規則

### 錄好的原始影片

- 先放到：[data/videos/raw_sequences](./data/videos/raw_sequences)

### 審片後合格影片

- 移到：[data/videos/approved_sequences](./data/videos/approved_sequences)

### 需要重錄的影片

- 移到：[data/videos/rerecord_needed](./data/videos/rerecord_needed)

### 標註相關 CSV

- 全部放在：[data/annotations](./data/annotations)

## 這個 repo 不放什麼

以下內容不應直接進這個 repo：

- 大量原始 gloss 影片
- `temp/` 原始資料池
- `data/landmarks/` 快取
- HST-SLR 實驗快取與中間產物
- 大量重複 comparison 影片

## Repo 結構

```text
docs/
  TEAM_TASKS.md
  FILE_GUIDE.md

data/
  annotations/
  videos/
    raw_sequences/
    approved_sequences/
    rerecord_needed/

baseline_gru/
  README.md
  scripts/
  models/
  results/
```
