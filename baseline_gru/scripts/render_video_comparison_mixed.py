from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MSG_DATA_ROOT", REPO_ROOT))
VIDEO_PATH = DATA_ROOT / "temp" / "Video Project.mp4"
ANNOTATION_PATH = DATA_ROOT / "data" / "annotations" / "video_project_time_annotations.csv"
WINDOW_PRED_PATH = BASELINE_DIR / "results" / "video_project_gru_mixed_window_predictions.csv"
MERGED_PRED_PATH = BASELINE_DIR / "results" / "video_project_gru_mixed_sequence_prediction.csv"
OUTPUT_PATH = BASELINE_DIR / "results" / "video_project_gru_mixed_comparison_clear.mp4"
HAND_MODEL = BASELINE_DIR / "models" / "hand_landmarker.task"
POSE_MODEL = BASELINE_DIR / "models" / "pose_landmarker.task"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_TITLE = load_font(24)
FONT_MAIN = load_font(20)
FONT_SMALL = load_font(16)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], font, color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def draw_landmarks(frame: np.ndarray, hand_result, pose_result) -> np.ndarray:
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


def current_gt(rows: list[dict[str, str]], sec: float) -> dict[str, str] | None:
    for row in rows:
        if float(row["start_sec"]) <= sec <= float(row["end_sec"]):
            return row
    return None


def current_pred(rows: list[dict[str, str]], sec: float) -> dict[str, str] | None:
    for row in rows:
        if float(row["start_sec"]) <= sec <= float(row["end_sec"]):
            return row
    return None


def merged_so_far(rows: list[dict[str, str]], sec: float) -> list[dict[str, str]]:
    return [row for row in rows if float(row["start_sec"]) <= sec]


def gt_progress(rows: list[dict[str, str]], sec: float) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    completed: list[dict[str, str]] = []
    active: dict[str, str] | None = None
    for row in rows:
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        if sec >= end_sec:
            completed.append(row)
        elif start_sec <= sec <= end_sec:
            active = row
            break
        else:
            break
    return completed, active


def draw_panel(
    base_frame: np.ndarray,
    sec: float,
    gt_rows: list[dict[str, str]],
    pred_row: dict[str, str] | None,
    merged_rows: list[dict[str, str]],
) -> np.ndarray:
    video_h, video_w = base_frame.shape[:2]
    panel_w = 760
    canvas = np.zeros((video_h, video_w + panel_w, 3), dtype=np.uint8)
    canvas[:, :video_w] = base_frame
    canvas[:, video_w:] = (18, 18, 18)

    canvas = draw_text(canvas, "Mixed Model Comparison", (video_w + 20, 18), FONT_TITLE, (255, 255, 255))
    canvas = draw_text(canvas, f"Time: {sec:05.2f}s", (video_w + 20, 52), FONT_MAIN, (220, 220, 220))

    completed_gt, active_gt = gt_progress(gt_rows, sec)
    gt_label = active_gt["label"] if active_gt else (completed_gt[-1]["label"] if completed_gt else "WAITING")
    gt_range = f"{active_gt['start_sec']} - {active_gt['end_sec']}" if active_gt else "-"
    canvas = draw_text(canvas, f"Current GT: {gt_label}", (video_w + 20, 88), FONT_MAIN, (255, 220, 0))
    canvas = draw_text(canvas, f"GT Range: {gt_range}", (video_w + 20, 116), FONT_SMALL, (220, 220, 220))

    pred_label = pred_row.get("smoothed_label", pred_row.get("predicted_label")) if pred_row else "WAITING"
    pred_score = pred_row.get("smoothed_score", pred_row.get("confidence")) if pred_row else "0.0"
    canvas = draw_text(canvas, f"Current Pred: {pred_label}", (video_w + 360, 88), FONT_MAIN, (0, 255, 0))
    canvas = draw_text(canvas, f"Pred Conf: {pred_score}", (video_w + 360, 116), FONT_SMALL, (220, 220, 220))

    left_x = video_w + 20
    right_x = video_w + 380
    header_y = 160
    canvas = draw_text(canvas, "Ground Truth Timeline", (left_x, header_y), FONT_MAIN, (255, 220, 0))
    canvas = draw_text(canvas, "Predicted Timeline", (right_x, header_y), FONT_MAIN, (0, 200, 255))

    list_y = header_y + 34
    visible_rows = 17
    for idx, row in enumerate(gt_rows[:visible_rows]):
        color = (120, 120, 120)
        if idx < len(completed_gt):
            color = (245, 245, 245)
        if active_gt is not None and row["order"] == active_gt["order"]:
            color = (255, 220, 0)
        text = f"{int(row['order']):02d}. {row['label']} [{row['start_sec']}-{row['end_sec']}]"
        canvas = draw_text(canvas, text, (left_x, list_y + idx * 24), FONT_SMALL, color)

    pred_tail = merged_rows[-visible_rows:]
    for idx, row in enumerate(pred_tail):
        color = (180, 240, 255)
        if idx == len(pred_tail) - 1:
            color = (0, 255, 255)
        text = f"{idx + 1:02d}. {row['predicted_label']} [{row['start_sec']}-{row['end_sec']}]"
        canvas = draw_text(canvas, text, (right_x, list_y + idx * 24), FONT_SMALL, color)

    footer1 = "GT stays fixed and lights up row by row."
    footer2 = "Prediction grows cumulatively instead of flashing."
    canvas = draw_text(canvas, footer1, (video_w + 20, video_h - 50), FONT_SMALL, (170, 170, 170))
    canvas = draw_text(canvas, footer2, (video_w + 20, video_h - 26), FONT_SMALL, (170, 170, 170))
    return canvas


def main() -> None:
    gt_rows = load_csv_rows(ANNOTATION_PATH)
    gt_rows = [row for row in gt_rows if row.get("start_sec") and row.get("end_sec")]
    pred_rows = load_csv_rows(WINDOW_PRED_PATH)
    merged_rows = load_csv_rows(MERGED_PRED_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_size = (src_w + 760, src_h)
    writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    frame_index = 0
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sec = frame_index / max(fps, 1e-6)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            visual = draw_landmarks(frame, hand_result, pose_result)
            output_frame = draw_panel(
                visual,
                sec,
                gt_rows,
                current_pred(pred_rows, sec),
                merged_so_far(merged_rows, sec),
            )
            writer.write(output_frame)
            frame_index += 1

    cap.release()
    writer.release()
    print(f"Saved mixed comparison video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
