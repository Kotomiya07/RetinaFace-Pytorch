"""
RetinaFace用のユーティリティ関数
"""

import cv2
import numpy as np
from typing import Optional


def load_image(image_path: str) -> np.ndarray:
    """
    画像を読み込み

    Args:
        image_path: 画像ファイルのパス

    Returns:
        読み込まれた画像（BGR形式）

    Raises:
        ValueError: 画像を読み込めない場合
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    return image


def draw_detection_results(
    image: np.ndarray,
    faces: list[dict],
    show_landmarks: bool = True,
    vis_threshold: float = 0.6,
    bbox_color: tuple[int, int, int] = (0, 0, 255),
    text_color: tuple[int, int, int] = (255, 255, 255),
    landmark_colors: Optional[list[tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    検出結果を画像に描画

    Args:
        image: 元画像
        faces: 検出結果のリスト
        show_landmarks: ランドマークを表示するかどうか
        vis_threshold: 表示する信頼度の閾値
        bbox_color: バウンディングボックスの色 (B, G, R)
        text_color: テキストの色 (B, G, R)
        landmark_colors: ランドマークの色のリスト

    Returns:
        描画後の画像
    """
    if landmark_colors is None:
        landmark_colors = [
            (0, 0, 255),  # 赤 - 右目
            (0, 255, 255),  # 黄 - 左目
            (255, 0, 255),  # マゼンタ - 鼻
            (0, 255, 0),  # 緑 - 右口角
            (255, 0, 0),  # 青 - 左口角
        ]

    img_draw = image.copy()

    for face in faces:
        if face["confidence"] < vis_threshold:
            continue

        # バウンディングボックスを描画
        bbox = face["bbox"]
        cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

        # 信頼度を表示
        text = f"{face['confidence']:.4f}"
        cv2.putText(img_draw, text, (bbox[0], bbox[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        # ランドマークを描画
        if show_landmarks and "landmarks" in face:
            landmarks = face["landmarks"]
            for i, (x, y) in enumerate(landmarks):
                color = landmark_colors[i] if i < len(landmark_colors) else (255, 255, 255)
                cv2.circle(img_draw, (int(x), int(y)), 1, color, 4)

    return img_draw


def crop_face(image: np.ndarray, bbox: list[int], margin_ratio: float = 0.2) -> np.ndarray:
    """
    バウンディングボックスから顔領域を切り出し

    Args:
        image: 元画像
        bbox: バウンディングボックス [x1, y1, x2, y2]
        margin_ratio: マージンの比率

    Returns:
        切り出された顔画像
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # マージンを追加
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    # 画像の境界内に収める
    h, w = image.shape[:2]
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, target_size: tuple[int, int], keep_aspect_ratio: bool = True) -> tuple[np.ndarray, float]:
    """
    画像をリサイズ

    Args:
        image: 元画像
        target_size: 目標サイズ (width, height)
        keep_aspect_ratio: アスペクト比を維持するかどうか

    Returns:
        リサイズされた画像とスケール比
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # パディングを追加
        if new_w < target_w or new_h < target_h:
            padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            padded[start_y : start_y + new_h, start_x : start_x + new_w] = resized
            resized = padded
    else:
        scale = 1.0
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized, scale


def calculate_face_area(bbox: list[int]) -> int:
    """
    顔の面積を計算

    Args:
        bbox: バウンディングボックス [x1, y1, x2, y2]

    Returns:
        面積（ピクセル数）
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def filter_faces_by_size(faces: list[dict], min_area: int = 100, max_area: Optional[int] = None) -> list[dict]:
    """
    顔のサイズでフィルタリング

    Args:
        faces: 検出結果のリスト
        min_area: 最小面積
        max_area: 最大面積（Noneの場合は制限なし）

    Returns:
        フィルタリングされた検出結果
    """
    filtered_faces = []

    for face in faces:
        area = calculate_face_area(face["bbox"])

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        filtered_faces.append(face)

    return filtered_faces


def sort_faces_by_confidence(faces: list[dict], descending: bool = True) -> list[dict]:
    """
    信頼度で顔をソート

    Args:
        faces: 検出結果のリスト
        descending: 降順でソートするかどうか

    Returns:
        ソートされた検出結果
    """
    return sorted(faces, key=lambda x: x["confidence"], reverse=descending)


def get_face_center(bbox: list[int]) -> tuple[int, int]:
    """
    顔の中心座標を取得

    Args:
        bbox: バウンディングボックス [x1, y1, x2, y2]

    Returns:
        中心座標 (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y
