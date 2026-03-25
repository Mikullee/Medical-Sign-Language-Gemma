from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MSG_DATA_ROOT", REPO_ROOT))
TEMP_DIR = DATA_ROOT / "temp"
GLOSS_PLUS_DIR = TEMP_DIR / "GLOSS+"
SEQ_DIR = TEMP_DIR / "SEQ"
CURATED_GLOSS_DIR = DATA_ROOT / "data" / "videos" / "curated_gloss"
LANDMARK_DIR = DATA_ROOT / "data" / "landmarks" / "mixed"
RESULTS_DIR = BASELINE_DIR / "results"
ANNOTATION_DIR = DATA_ROOT / "data" / "annotations"
HAND_MODEL = BASELINE_DIR / "models" / "hand_landmarker.task"
POSE_MODEL = BASELINE_DIR / "models" / "pose_landmarker.task"
SEQ_ANNOTATION_PATH = SEQ_DIR / "sequence_annotations_all_with_blank.csv"

VIDEO_SUFFIXES = {".mp4", ".mov"}
SEED = 42
FRAME_STEP = 2
WINDOW_SIZE = 16
TRAIN_WINDOW_STRIDE = 8
PREDICT_WINDOW_STRIDE = 4
MIN_CONFIDENCE = 0.35
BLANK_LABEL = "BLANK"
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

POSE_SIZE = 33 * 3
HAND_SIZE = 21 * 3
GEOMETRY_FEATURE_DIM = 18
BASE_FEATURE_DIM = POSE_SIZE + HAND_SIZE * 2 + GEOMETRY_FEATURE_DIM
FEATURE_DIM = BASE_FEATURE_DIM * 2
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

MODEL_PATH = RESULTS_DIR / "gloss_gru_mixed_model.pt"
METRICS_PATH = RESULTS_DIR / "gru_mixed_training_metrics.json"
WINDOW_PRED_PATH = RESULTS_DIR / "video_project_gru_mixed_window_predictions.csv"
MERGED_PRED_PATH = RESULTS_DIR / "video_project_gru_mixed_sequence_prediction.csv"
DATASET_SUMMARY_PATH = RESULTS_DIR / "gru_mixed_training_dataset_summary.csv"
HISTORY_PATH = RESULTS_DIR / "gru_mixed_training_history.csv"
LABEL_COUNTS_PATH = RESULTS_DIR / "gru_mixed_training_label_counts.csv"
LABELS_PATH = ANNOTATION_DIR / "gru_mixed_available_gloss_labels.csv"

CANONICAL_LABELS = [
    "冷氣團",
    "東北部",
    "越來越",
    "地區",
    "早晚",
    "保暖",
    "做好",
    "今天",
    "大陸",
    "北部",
    "大家",
    "記得",
    "其他",
    "晚",
    "冷",
    "歸",
    "要",
]


@dataclass
class VideoFrames:
    path: Path
    label: str
    fps: float
    frames: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"^\d+[._]?", "", name)
    name = name.replace("（", "(").replace("）", ")")
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"_\d+(?:_\d+)?$", "", name)
    name = re.sub(r"[-_](?:\d+)$", "", name)
    name = re.sub(r"\s+", "", name)
    name = name.strip("._- ")
    return name


def normalize_label_from_name(name: str) -> str:
    cleaned = clean_name(name)

    synonym_groups = {
        "保暖": ["保暖", "穿衣"],
        "做好": ["做好", "成立"],
        "地區": ["地區", "地"],
        "早晚": ["早晚", "早"],
        "冷氣團": ["冷氣團", "冷氣"],
        "越來越": ["越來越", "越"],
        "歸": ["歸", "回家"],
    }
    for canonical, options in synonym_groups.items():
        if any(option in cleaned for option in options):
            return canonical

    for label in CANONICAL_LABELS:
        if label in cleaned:
            return label
    return cleaned


def cache_name_for(path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}.npz"


def extract_frame_vector(hand_result, pose_result) -> np.ndarray:
    values: list[float] = []
    pose_landmarks = getattr(pose_result, "pose_landmarks", [])
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            values.extend([lm.x, lm.y, lm.z])
    else:
        values.extend([0.0] * POSE_SIZE)

    left = [0.0] * HAND_SIZE
    right = [0.0] * HAND_SIZE
    handedness_list = getattr(hand_result, "handedness", [])
    hand_landmarks_list = getattr(hand_result, "hand_landmarks", [])
    for handedness, landmarks in zip(handedness_list, hand_landmarks_list):
        flat = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        if handedness[0].category_name.lower() == "left":
            left = flat
        else:
            right = flat

    values.extend(left)
    values.extend(right)
    return np.array(values, dtype=np.float32)


def extract_video_frames(video_path: Path, label: str) -> VideoFrames:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_vectors: list[np.ndarray] = []

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % FRAME_STEP != 0:
                idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)
            frame_vectors.append(extract_frame_vector(hand_result, pose_result))
            idx += 1

    cap.release()
    if not frame_vectors:
        raise RuntimeError(f"No landmarks extracted from {video_path}")

    frames = np.stack(frame_vectors)
    cache_path = LANDMARK_DIR / cache_name_for(video_path)
    np.savez_compressed(cache_path, fps=fps, frames=frames, label=label)
    return VideoFrames(path=video_path, label=label, fps=float(fps), frames=frames)


def load_or_extract_video_frames(video_path: Path, label: str) -> VideoFrames:
    cache_path = LANDMARK_DIR / cache_name_for(video_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return VideoFrames(path=video_path, label=label, fps=float(data["fps"]), frames=data["frames"].astype(np.float32))
    return extract_video_frames(video_path, label)


def normalize_relative_frames(frames: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        pose = frame[:POSE_SIZE].reshape(33, 3).copy()
        left = frame[POSE_SIZE:POSE_SIZE + HAND_SIZE].reshape(21, 3).copy()
        right = frame[POSE_SIZE + HAND_SIZE:].reshape(21, 3).copy()

        left_shoulder = pose[LEFT_SHOULDER]
        right_shoulder = pose[RIGHT_SHOULDER]
        valid_left = np.any(left_shoulder)
        valid_right = np.any(right_shoulder)

        if valid_left and valid_right:
            center = (left_shoulder + right_shoulder) / 2.0
            scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        else:
            valid_pose = pose[np.any(pose != 0, axis=1)]
            if len(valid_pose):
                center = valid_pose.mean(axis=0)
                scale = float(np.linalg.norm(np.ptp(valid_pose[:, :2], axis=0)))
            else:
                center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                scale = 1.0

        scale = float(max(scale, 1e-3))
        for arr in (pose, left, right):
            valid_mask = np.any(arr != 0, axis=1)
            arr[valid_mask] = (arr[valid_mask] - center) / scale

        normalized[i] = np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)]).astype(np.float32)
    return normalized


def compute_geometry_features(frame: np.ndarray) -> np.ndarray:
    pose = frame[:POSE_SIZE].reshape(33, 3)
    left = frame[POSE_SIZE:POSE_SIZE + HAND_SIZE].reshape(21, 3)
    right = frame[POSE_SIZE + HAND_SIZE:].reshape(21, 3)

    def valid_rows(arr: np.ndarray) -> np.ndarray:
        return arr[np.any(arr != 0, axis=1)]

    def center_or_zero(arr: np.ndarray) -> np.ndarray:
        valid = valid_rows(arr)
        return valid.mean(axis=0) if len(valid) else np.zeros(3, dtype=np.float32)

    def point_or_zero(arr: np.ndarray, idx: int) -> np.ndarray:
        point = arr[idx]
        return point if np.any(point != 0) else np.zeros(3, dtype=np.float32)

    left_center = center_or_zero(left)
    right_center = center_or_zero(right)
    left_wrist = point_or_zero(pose, LEFT_WRIST)
    right_wrist = point_or_zero(pose, RIGHT_WRIST)
    nose = point_or_zero(pose, NOSE)
    left_shoulder = point_or_zero(pose, LEFT_SHOULDER)
    right_shoulder = point_or_zero(pose, RIGHT_SHOULDER)

    def distance(a: np.ndarray, b: np.ndarray) -> float:
        if not np.any(a) or not np.any(b):
            return 0.0
        return float(np.linalg.norm(a - b))

    return np.array(
        [
            *left_center.tolist(),
            *right_center.tolist(),
            *left_wrist.tolist(),
            *right_wrist.tolist(),
            distance(left_center, right_center),
            distance(left_wrist, right_wrist),
            distance(left_center, nose),
            distance(right_center, nose),
            distance(left_wrist, left_shoulder),
            distance(right_wrist, right_shoulder),
        ],
        dtype=np.float32,
    )


def enrich_frame_features(frames: np.ndarray) -> np.ndarray:
    geometry = np.stack([compute_geometry_features(frame) for frame in frames]).astype(np.float32)
    return np.concatenate([frames, geometry], axis=1).astype(np.float32)


def add_velocity_features(frames: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(frames, dtype=np.float32)
    if len(frames) > 1:
        velocity[1:] = frames[1:] - frames[:-1]
    return np.concatenate([frames, velocity], axis=1).astype(np.float32)


def build_windows(frames: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    if len(frames) == 0:
        return []
    if len(frames) < window_size:
        pad_count = window_size - len(frames)
        padded = np.concatenate([frames, np.repeat(frames[-1:, :], pad_count, axis=0)], axis=0)
        return [padded]

    windows = []
    for start in range(0, len(frames) - window_size + 1, stride):
        windows.append(frames[start:start + window_size])
    if (len(frames) - window_size) % stride != 0:
        windows.append(frames[-window_size:])
    return windows


def display_relative_path(path: Path) -> str:
    for base in (TEMP_DIR, ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def list_gloss_training_files() -> list[Path]:
    files: list[Path] = []
    if CURATED_GLOSS_DIR.exists():
        for path in sorted(CURATED_GLOSS_DIR.glob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                files.append(path)

    if GLOSS_PLUS_DIR.exists():
        for path in sorted(GLOSS_PLUS_DIR.glob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                files.append(path)
    return files


def split_gloss_files_by_label(files: list[Path]) -> tuple[list[Path], list[Path]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[normalize_label_from_name(path.stem)].append(path)

    train_files: list[Path] = []
    val_files: list[Path] = []
    for _label, items in sorted(grouped.items()):
        items = items[:]
        rng.shuffle(items)
        if len(items) <= 2:
            train_files.extend(items)
            continue
        val_count = max(1, round(len(items) * 0.2))
        val_files.extend(items[:val_count])
        train_files.extend(items[val_count:])
    return sorted(train_files), sorted(val_files)


def load_sequence_annotations() -> pd.DataFrame:
    if not SEQ_ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Missing sequence annotations: {SEQ_ANNOTATION_PATH}")
    df = pd.read_csv(SEQ_ANNOTATION_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["video_name", "start_sec", "end_sec", "label"]).copy()
    df["label"] = df["label"].astype(str).map(normalize_label_from_name)
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)
    return df


def split_sequence_video_names(video_names: list[str]) -> tuple[set[str], set[str]]:
    rng = random.Random(SEED)
    names = sorted(set(video_names))
    rng.shuffle(names)
    if len(names) <= 4:
        return set(names), set()
    val_count = max(2, round(len(names) * 0.2))
    val_names = set(names[:val_count])
    train_names = set(names[val_count:])
    return train_names, val_names


class SequenceDataset(Dataset):
    def __init__(self, windows: list[np.ndarray], labels: list[int]) -> None:
        self.windows = [torch.tensor(w, dtype=torch.float32) for w in windows]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float, bidirectional: bool = True) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _output, hidden = self.gru(x)
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        return self.head(self.dropout(last_hidden))


def make_windows_for_gloss_files(
    files: list[Path],
    label_to_idx: dict[str, int],
    split_name: str,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict] = []

    for path in files:
        label = normalize_label_from_name(path.stem)
        video = load_or_extract_video_frames(path, label)
        rel_frames = normalize_relative_frames(video.frames)
        enriched = enrich_frame_features(rel_frames)
        seq_frames = add_velocity_features(enriched)
        clip_windows = build_windows(seq_frames, WINDOW_SIZE, TRAIN_WINDOW_STRIDE)
        for window in clip_windows:
            windows.append(window)
            labels.append(label_to_idx[label])
        rows.append(
            {
                "source": "gloss",
                "relative_path": display_relative_path(path),
                "file_name": path.name,
                "label": label,
                "num_frames": len(video.frames),
                "num_windows": len(clip_windows),
                "split": split_name,
            }
        )
        print(f"[gloss/{split_name}] {path.name} -> {label} ({len(clip_windows)} windows)")
    return windows, labels, rows


def make_windows_for_sequence_annotations(
    annotation_df: pd.DataFrame,
    label_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int], list[dict]]:
    train_windows: list[np.ndarray] = []
    train_labels: list[int] = []
    val_windows: list[np.ndarray] = []
    val_labels: list[int] = []
    rows: list[dict] = []

    train_video_names, val_video_names = split_sequence_video_names(annotation_df["video_name"].tolist())
    grouped = annotation_df.groupby("video_name", sort=True)
    for video_name, group in grouped:
        video_path = SEQ_DIR / video_name
        if not video_path.exists():
            continue
        video = load_or_extract_video_frames(video_path, "SEQUENCE")
        frame_seconds = FRAME_STEP / max(video.fps, 1e-6)
        rel_frames = normalize_relative_frames(video.frames)
        enriched = enrich_frame_features(rel_frames)
        seq_frames = add_velocity_features(enriched)
        split_name = "val" if video_name in val_video_names else "train"

        for _, row in group.iterrows():
            label = row["label"]
            if label not in label_to_idx:
                continue
            start_idx = max(0, int(np.floor(float(row["start_sec"]) / frame_seconds)))
            end_idx = max(start_idx + 1, int(np.ceil(float(row["end_sec"]) / frame_seconds)))
            end_idx = min(end_idx, len(seq_frames))
            segment_frames = seq_frames[start_idx:end_idx]
            if len(segment_frames) == 0:
                continue
            clip_windows = build_windows(segment_frames, WINDOW_SIZE, TRAIN_WINDOW_STRIDE)
            target_windows = val_windows if split_name == "val" else train_windows
            target_labels = val_labels if split_name == "val" else train_labels
            for window in clip_windows:
                target_windows.append(window)
                target_labels.append(label_to_idx[label])
            rows.append(
                {
                    "source": "sequence",
                    "relative_path": display_relative_path(video_path),
                    "file_name": video_name,
                    "label": label,
                    "segment_start_sec": round(float(row["start_sec"]), 3),
                    "segment_end_sec": round(float(row["end_sec"]), 3),
                    "num_frames": len(segment_frames),
                    "num_windows": len(clip_windows),
                    "split": split_name,
                }
            )
            print(
                f"[sequence/{split_name}] {video_name} {label} "
                f"[{row['start_sec']:.2f}-{row['end_sec']:.2f}] -> {len(clip_windows)} windows"
            )
    return train_windows, train_labels, val_windows, val_labels, rows


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(x)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def smooth_window_predictions(rows: list[dict], radius: int = 1) -> list[dict]:
    smoothed: list[dict] = []
    for idx, row in enumerate(rows):
        scores: dict[str, float] = defaultdict(float)
        for j in range(max(0, idx - radius), min(len(rows), idx + radius + 1)):
            neighbor = rows[j]
            label = neighbor["predicted_label"]
            if label in {"UNKNOWN", BLANK_LABEL}:
                continue
            scores[label] += float(neighbor["confidence"]) / (1 + abs(idx - j))
        best_label = row["predicted_label"]
        best_score = 0.0
        if scores:
            best_label, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score < MIN_CONFIDENCE:
            best_label = "UNKNOWN"
        updated = dict(row)
        updated["smoothed_label"] = best_label
        updated["smoothed_score"] = round(float(best_score), 4)
        smoothed.append(updated)
    return smoothed


def merge_predictions(rows: list[dict], min_run: int = 2, min_duration_sec: float = 0.45) -> list[dict]:
    merged: list[dict] = []
    i = 0
    while i < len(rows):
        label = rows[i].get("smoothed_label", rows[i]["predicted_label"])
        start_idx = i
        max_conf = float(rows[i]["confidence"])
        while i + 1 < len(rows) and rows[i + 1].get("smoothed_label", rows[i + 1]["predicted_label"]) == label:
            i += 1
            max_conf = max(max_conf, float(rows[i]["confidence"]))

        run_rows = rows[start_idx:i + 1]
        start_sec = float(run_rows[0]["start_sec"])
        end_sec = float(run_rows[-1]["end_sec"])
        duration = end_sec - start_sec
        if label not in {"UNKNOWN", BLANK_LABEL} and len(run_rows) >= min_run and duration >= min_duration_sec:
            if merged and merged[-1]["predicted_label"] == label and start_sec <= float(merged[-1]["end_sec"]) + 0.35:
                merged[-1]["end_sec"] = round(end_sec, 2)
                merged[-1]["confidence"] = round(max(float(merged[-1]["confidence"]), max_conf), 4)
            else:
                merged.append(
                    {
                        "predicted_label": label,
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "confidence": round(max_conf, 4),
                    }
                )
        i += 1
    return merged


def predict_target(model: nn.Module, label_names: list[str], device: torch.device) -> None:
    target_file = TEMP_DIR / "Video Project.mp4"
    target_frames = load_or_extract_video_frames(target_file, "VIDEO_PROJECT")
    rel_frames = normalize_relative_frames(target_frames.frames)
    enriched_frames = enrich_frame_features(rel_frames)
    seq_frames = add_velocity_features(enriched_frames)
    windows = build_windows(seq_frames, WINDOW_SIZE, PREDICT_WINDOW_STRIDE)
    x = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    frame_seconds = FRAME_STEP / max(target_frames.fps, 1e-6)
    rows = []
    for i, prob in enumerate(probs):
        pred_idx = int(np.argmax(prob))
        confidence = float(prob[pred_idx])
        start_sec = round(i * PREDICT_WINDOW_STRIDE * frame_seconds, 2)
        end_sec = round(start_sec + WINDOW_SIZE * frame_seconds, 2)
        rows.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "predicted_label": label_names[pred_idx] if confidence >= MIN_CONFIDENCE else "UNKNOWN",
                "confidence": round(confidence, 4),
            }
        )

    smoothed = smooth_window_predictions(rows)
    merged = merge_predictions(smoothed)
    pd.DataFrame(smoothed).to_csv(WINDOW_PRED_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(merged).to_csv(MERGED_PRED_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    set_seed(SEED)
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    gloss_files = list_gloss_training_files()
    seq_df = load_sequence_annotations()
    labels = sorted({normalize_label_from_name(path.stem) for path in gloss_files} | set(seq_df["label"].tolist()))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = [label for label, _idx in sorted(label_to_idx.items(), key=lambda item: item[1])]

    gloss_train_files, gloss_val_files = split_gloss_files_by_label(gloss_files)
    gloss_train_windows, gloss_train_labels, gloss_train_rows = make_windows_for_gloss_files(gloss_train_files, label_to_idx, "train")
    gloss_val_windows, gloss_val_labels, gloss_val_rows = make_windows_for_gloss_files(gloss_val_files, label_to_idx, "val")
    seq_train_windows, seq_train_labels, seq_val_windows, seq_val_labels, seq_rows = make_windows_for_sequence_annotations(seq_df, label_to_idx)

    train_windows = gloss_train_windows + seq_train_windows
    train_labels = gloss_train_labels + seq_train_labels
    val_windows = gloss_val_windows + seq_val_windows
    val_labels = gloss_val_labels + seq_val_labels

    if not train_windows or not val_windows:
        raise RuntimeError("Mixed training dataset is empty after loading gloss and sequence data")

    pd.DataFrame(gloss_train_rows + gloss_val_rows + seq_rows).to_csv(DATASET_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"label": label, "index": idx} for label, idx in label_to_idx.items()]).sort_values("index").to_csv(
        LABELS_PATH, index=False, encoding="utf-8-sig"
    )

    train_dataset = SequenceDataset(train_windows, train_labels)
    val_dataset = SequenceDataset(val_windows, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(labels),
        dropout=DROPOUT,
        bidirectional=True,
    ).to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(labels)), y=np.array(train_labels))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x)
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_total += len(x)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
        )
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "labels": idx_to_label,
                "config": {
                    "frame_step": FRAME_STEP,
                    "window_size": WINDOW_SIZE,
                    "window_stride": TRAIN_WINDOW_STRIDE,
                    "predict_window_stride": PREDICT_WINDOW_STRIDE,
                    "min_confidence": MIN_CONFIDENCE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "bidirectional": True,
                    "feature_mode": "relative+geometry+velocity",
                    "base_feature_dim": BASE_FEATURE_DIM,
                    "feature_dim": FEATURE_DIM,
                    "training_sources": ["curated_gloss", "GLOSS+", "SEQ_segments_with_blank"],
                },
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model")

    torch.save(best_state, MODEL_PATH)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    (
        pd.Series([idx_to_label[idx] for idx in train_labels])
        .value_counts()
        .rename_axis("label")
        .reset_index(name="num_training_windows")
        .sort_values("label")
        .to_csv(LABEL_COUNTS_PATH, index=False, encoding="utf-8-sig")
    )

    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "num_labels": len(labels),
                "num_training_windows": len(train_labels),
                "num_validation_windows": len(val_labels),
                "best_val_accuracy": round(float(best_val_acc), 4),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "window_size": WINDOW_SIZE,
                "window_stride": TRAIN_WINDOW_STRIDE,
                "predict_window_stride": PREDICT_WINDOW_STRIDE,
                "frame_step": FRAME_STEP,
                "min_confidence": MIN_CONFIDENCE,
                "feature_mode": "relative+geometry+velocity",
                "training_sources": ["curated_gloss", "GLOSS+", "SEQ_segments_with_blank"],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    model.load_state_dict(best_state["model_state_dict"])
    predict_target(model, idx_to_label, device)

    print("[done] saved:")
    print(f"- {MODEL_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {HISTORY_PATH}")
    print(f"- {LABEL_COUNTS_PATH}")
    print(f"- {WINDOW_PRED_PATH}")
    print(f"- {MERGED_PRED_PATH}")


if __name__ == "__main__":
    main()
