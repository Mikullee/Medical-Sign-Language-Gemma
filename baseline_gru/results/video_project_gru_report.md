# Video Project GRU Summary

## Model

- Input: MediaPipe pose + hand landmarks
- Features:
  - relative coordinates
  - geometry features
  - velocity features
- Classifier: bidirectional GRU

## Two versions

### Best numeric version
- Validation accuracy: `0.7976`
- Focus: gloss-driven training
- Demo: `video_project_gru_comparison_clear.mp4`

### Mixed + sequence + blank version
- Validation accuracy: `0.7546`
- Focus: closer to continuous sign recognition
- Demo: `video_project_gru_mixed_comparison_clear.mp4`

## Interpretation

- Best numeric version currently gives the strongest window-level classification score.
- Mixed version is closer to the final project goal because it includes sequence supervision and transition handling.
- Current validation accuracy is computed on fixed windows, not on full-sequence perfect recognition.

## Next step

- Keep `GRU baseline` as the comparison baseline
- Expand sequence data and annotation quality
- Continue evaluating `HST-SLR` as the next-stage continuous model
