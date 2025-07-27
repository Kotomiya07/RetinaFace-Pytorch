"""
RetinaFace Face Detection Library

簡単に使用できる顔検出ライブラリです。
"""

try:
    from .detector import RetinaFaceDetector
    from .utils import load_image, draw_detection_results

    __version__ = "1.0.0"
    __all__ = ["RetinaFaceDetector", "load_image", "draw_detection_results"]

except ImportError as e:
    print(f"Warning: {e}")
    # 開発時のためのフォールバック
    __version__ = "1.0.0"
    __all__ = []
