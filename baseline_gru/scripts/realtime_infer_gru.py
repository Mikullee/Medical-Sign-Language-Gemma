from __future__ import annotations

import argparse
import os
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASELINE_DIR / "results"
MODEL_PATH = Path(os.environ.get("MSG_GRU_MODEL", str(BASELINE_DIR / "models" / "gloss_gru_model.pt")))
HAND_MODEL = BASELINE_DIR / "models" / "hand_landmarker.task"
POSE_MODEL = BASELINE_DIR / "models" / "pose_landmarker.task"

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
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
]


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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_LARGE = load_font(28)
FONT_MEDIUM = load_font(22)
FONT_SMALL = load_font(18)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], font, color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


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
        label = handedness[0].category_name.lower()
        flat = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        if label == "left":
            left = flat
        else:
            right = flat

    values.extend(left)
    values.extend(right)
    return np.array(values, dtype=np.float32)


def normalize_relative_frame(frame: np.ndarray) -> np.ndarray:
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

    return np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)]).astype(np.float32)


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

    geometry = np.array(
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
    return geometry


def enrich_frame_feature(frame: np.ndarray) -> np.ndarray:
    geometry = compute_geometry_features(frame)
    return np.concatenate([frame, geometry], axis=0).astype(np.float32)


def add_velocity_to_window(frames: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(frames, dtype=np.float32)
    if len(frames) > 1:
        velocity[1:] = frames[1:] - frames[:-1]
    return np.concatenate([frames, velocity], axis=1).astype(np.float32)


def choose_smoothed_label(history: deque[tuple[str, float]]) -> tuple[str, float]:
    if not history:
        return "WAITING", 0.0
    labels = [label for label, _ in history]
    most_common, _ = Counter(labels).most_common(1)[0]
    confidences = [conf for label, conf in history if label == most_common]
    return most_common, float(sum(confidences) / max(len(confidences), 1))


def draw_landmark_points(frame: np.ndarray, hand_result, pose_result) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    pose_landmarks = getattr(pose_result, "pose_landmarks", []) if pose_result is not None else []
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

    handedness_list = getattr(hand_result, "handedness", []) if hand_result is not None else []
    hand_landmarks_list = getattr(hand_result, "hand_landmarks", []) if hand_result is not None else []
    for handedness, landmarks in zip(handedness_list, hand_landmarks_list):
        color = (0, 255, 0) if handedness[0].category_name.lower() == "left" else (255, 0, 0)
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 4, color, -1)

        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for a, b in hand_connections:
            if a < len(points) and b < len(points):
                cv2.line(canvas, points[a], points[b], color, 2)

    return canvas


def draw_overlay(
    frame: np.ndarray,
    display_label: str,
    display_confidence: float,
    sequence_labels: list[str],
    fps_value: float,
    current_streak: int,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (10, 10), (520, 165), (0, 0, 0), -1)
    cv2.rectangle(canvas, (10, h - 58), (w - 10, h - 10), (0, 0, 0), -1)

    canvas = draw_text(canvas, f"Current: {display_label}", (25, 20), FONT_LARGE, (0, 255, 0))
    canvas = draw_text(canvas, f"Confidence: {display_confidence:.2f}", (25, 58), FONT_MEDIUM, (255, 255, 255))
    canvas = draw_text(canvas, f"Streak: {current_streak}", (25, 90), FONT_MEDIUM, (255, 255, 255))
    canvas = draw_text(canvas, f"FPS: {fps_value:.1f}", (25, 122), FONT_MEDIUM, (200, 200, 0))
    canvas = draw_text(canvas, "Quit: press Q or Esc", (25, 146), FONT_SMALL, (180, 180, 180))

    seq_text = "Sequence: " + (" ".join(sequence_labels[-8:]) if sequence_labels else "-")
    canvas = draw_text(canvas, seq_text, (25, h - 48), FONT_MEDIUM, (255, 255, 255))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime gloss inference using relative-coordinate GRU model.")
    parser.add_argument("--source", default="1", help="Camera index like 0/1/2 or a video path. Default 1 is often OBS Virtual Camera on Windows.")
    parser.add_argument("--backend", choices=["auto", "dshow"], default="dshow", help="VideoCapture backend on Windows.")
    parser.add_argument("--save-log", action="store_true", help="Save per-step predictions to CSV on exit.")
    parser.add_argument("--min-streak", type=int, default=3, help="How many consecutive stable predictions are required before appending to sequence.")
    parser.add_argument("--width", type=int, default=1280, help="Display width of the realtime window.")
    parser.add_argument("--height", type=int, default=900, help="Display height of the realtime window.")
    return parser.parse_args()


def open_capture(source: str, backend: str) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        if backend == "dshow":
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(source)


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    labels = checkpoint["labels"]
    config = checkpoint["config"]
    window_size = int(config["window_size"])
    frame_step = int(config["frame_step"])
    min_confidence = float(config.get("min_confidence", 0.45))

    model = GRUClassifier(
        input_dim=FEATURE_DIM,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_classes=len(labels),
        dropout=float(config["dropout"]),
        bidirectional=bool(config.get("bidirectional", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Opening video source: {args.source} (backend={args.backend})")
    print("If OBS Virtual Camera is not detected, try --source 0 or --source 2.")
    print("Press Q or Esc to exit realtime recognition.")
    cap = open_capture(str(args.source), args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    window_name = "Realtime Gloss Recognition (GRU)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
    prediction_history: deque[tuple[str, float]] = deque(maxlen=5)
    merged_sequence: list[str] = []
    prediction_log: list[dict] = []
    last_hand_result = None
    last_pose_result = None
    frame_index = 0
    last_tick = cv2.getTickCount()
    fps_value = 0.0
    stable_label = "BUFFERING"
    stable_streak = 0

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_result = hand_landmarker.detect(mp_image)
                pose_result = pose_landmarker.detect(mp_image)
                last_hand_result = hand_result
                last_pose_result = pose_result

                frame_vector = extract_frame_vector(hand_result, pose_result)
                normalized_frame = normalize_relative_frame(frame_vector)
                frame_buffer.append(enrich_frame_feature(normalized_frame))

                if len(frame_buffer) == window_size:
                    seq_array = add_velocity_to_window(np.stack(frame_buffer))
                    seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        probs = torch.softmax(model(seq), dim=1)[0].cpu().numpy()

                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                    predicted_label = str(labels[pred_idx]) if confidence >= min_confidence else "UNKNOWN"
                    prediction_history.append((predicted_label, confidence))
                    display_label, display_confidence = choose_smoothed_label(prediction_history)

                    if display_label == stable_label:
                        stable_streak += 1
                    else:
                        stable_label = display_label
                        stable_streak = 1

                    if display_label != "UNKNOWN" and stable_streak >= args.min_streak:
                        if not merged_sequence or merged_sequence[-1] != display_label:
                            merged_sequence.append(display_label)

                    prediction_log.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "frame_index": frame_index,
                            "predicted_label": predicted_label,
                            "confidence": round(confidence, 4),
                            "display_label": display_label,
                            "display_confidence": round(display_confidence, 4),
                            "stable_streak": stable_streak,
                        }
                    )
                else:
                    display_label, display_confidence = "BUFFERING", 0.0
                    stable_label = display_label
                    stable_streak = 0
            else:
                display_label, display_confidence = choose_smoothed_label(prediction_history)

            now_tick = cv2.getTickCount()
            elapsed = (now_tick - last_tick) / cv2.getTickFrequency()
            if elapsed > 0:
                fps_value = 1.0 / elapsed
            last_tick = now_tick

            landmark_view = draw_landmark_points(frame, last_hand_result, last_pose_result)
            overlay = draw_overlay(landmark_view, display_label, display_confidence, merged_sequence, fps_value, stable_streak)
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if args.save_log and prediction_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"realtime_gru_predictions_{stamp}.csv"
        pd.DataFrame(prediction_log).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved realtime prediction log: {out_path}")

    print("Final merged sequence:", " ".join(merged_sequence) if merged_sequence else "-")


if __name__ == "__main__":
    main()
