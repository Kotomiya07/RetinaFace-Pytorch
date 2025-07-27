# RetinaFace Pytorch

RetinaFaceを使いやすいライブラリとして実装したPythonパッケージです。

## 特徴

- **簡単なAPI**: 数行のコードで顔検出が可能
- **高精度**: RetinaFaceアーキテクチャによる高精度な顔検出
- **ランドマーク検出**: 顔の特徴点（目、鼻、口角）も同時に検出
- **柔軟な入力**: 画像ファイル、numpy配列、動画ファイル、Webカメラに対応
- **複数モデル対応**: MobileNet 0.25（軽量・高速）とResNet50（高精度）を選択可能

## インストール

```bash
# 必要な依存関係をインストール
uv sync

# 重みファイルをダウンロード（必要に応じて）
# ./weights/ ディレクトリに以下のファイルを配置:
# - mobilenet0.25.pth (MobileNet用)
# - resnet50.pth (ResNet50用)
```

## 基本的な使用方法

### 1. 基本的な顔検出

```python
from retinaface import RetinaFaceDetector

# 検出器を初期化
detector = RetinaFaceDetector(
    network='resnet50',  # または 'mobile0.25'
    confidence_threshold=0.8
)

# 顔検出を実行
faces = detector.detect('path/to/image.jpg', return_landmarks=True)

# 結果を表示
for i, face in enumerate(faces):
    print(f"顔 {i+1}:")
    print(f"  バウンディングボックス: {face['bbox']}")
    print(f"  信頼度: {face['confidence']:.4f}")
    if 'landmarks' in face:
        print(f"  ランドマーク: {face['landmarks']}")
```

### 2. 検出結果の可視化

```python
from retinaface import RetinaFaceDetector, load_image, draw_detection_results
import cv2

# 検出器を初期化
detector = RetinaFaceDetector(network='resnet50')

# 画像を読み込み
image = load_image('path/to/image.jpg')

# 顔検出
faces = detector.detect(image, return_landmarks=True)

# 結果を画像に描画
result_image = draw_detection_results(image, faces, show_landmarks=True)

# 結果を保存
cv2.imwrite('result.jpg', result_image)
```

### 3. 動画での顔検出

```python
detector = RetinaFaceDetector(network='mobile0.25')  # 高速化のため軽量モデルを使用

# 動画での顔検出
detector.detect_faces_in_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4',
    show_landmarks=True
)
```

### 4. Webカメラでのリアルタイム検出

```python
import cv2

detector = RetinaFaceDetector(network='mobile0.25')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 顔検出
    faces = detector.detect(frame, return_landmarks=True)
    
    # 結果を描画
    result_frame = detector.draw_detections(frame, faces)
    
    # 表示
    cv2.imshow('Face Detection', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## API リファレンス

### RetinaFaceDetector

メインの顔検出クラス

#### 初期化パラメータ

- `network` (str): 使用するネットワーク (`'mobile0.25'` または `'resnet50'`)
- `weights_path` (str, optional): 重みファイルのパス
- `device` (str): 使用するデバイス (`'auto'`, `'cpu'`, `'cuda'`)
- `confidence_threshold` (float): 信頼度の閾値 (デフォルト: 0.02)
- `nms_threshold` (float): NMSの閾値 (デフォルト: 0.4)
- `vis_threshold` (float): 可視化時の閾値 (デフォルト: 0.6)

#### メソッド

##### `detect(image, return_landmarks=True)`

顔検出を実行

**パラメータ:**
- `image`: 画像パス（str）またはnumpy配列
- `return_landmarks` (bool): ランドマークを返すかどうか

**戻り値:**
検出結果のリスト。各要素は以下のキーを持つ辞書:
- `'bbox'`: [x1, y1, x2, y2] バウンディングボックス
- `'confidence'`: 信頼度
- `'landmarks'`: (return_landmarks=Trueの場合) ランドマーク座標

##### `detect_faces_in_video(video_path, output_path=None, show_landmarks=True)`

動画内の顔検出を実行

##### `draw_detections(image, faces, show_landmarks=True)`

検出結果を画像に描画

### ユーティリティ関数

#### `load_image(image_path)`

画像を読み込み

#### `draw_detection_results(image, faces, show_landmarks=True, ...)`

検出結果を画像に描画（カスタマイズ可能）

#### `crop_face(image, bbox, margin_ratio=0.2)`

バウンディングボックスから顔領域を切り出し

#### `filter_faces_by_size(faces, min_area=100, max_area=None)`

顔のサイズでフィルタリング

#### `sort_faces_by_confidence(faces, descending=True)`

信頼度で顔をソート

## 設定例

### 高精度設定（ResNet50）

```python
detector = RetinaFaceDetector(
    network='resnet50',
    confidence_threshold=0.8,
    nms_threshold=0.4,
    vis_threshold=0.8
)
```

### 高速設定（MobileNet 0.25）

```python
detector = RetinaFaceDetector(
    network='mobile0.25',
    confidence_threshold=0.7,
    nms_threshold=0.4,
    vis_threshold=0.7
)
```

### CPU使用設定

```python
detector = RetinaFaceDetector(
    network='mobile0.25',
    device='cpu',
    confidence_threshold=0.8
)
```

## 性能比較

| モデル | 速度 | 精度 | 用途 |
|--------|------|------|------|
| MobileNet 0.25 | 高速 | 標準 | リアルタイム処理、動画処理 |
| ResNet50 | 標準 | 高精度 | 高精度が必要な用途、静止画処理 |

## トラブルシューティング

### 重みファイルが見つからない

```
FileNotFoundError: 重みファイルが見つかりません
```

→ `./weights/` ディレクトリに適切な重みファイルを配置してください。

### CUDA関連のエラー

```
RuntimeError: CUDA out of memory
```

→ CPUモードを使用するか、バッチサイズを小さくしてください：

```python
detector = RetinaFaceDetector(device='cpu')
```

### OpenCV関連のエラー

```
ImportError: No module named 'cv2'
```

→ OpenCVをインストールしてください：

```bash
uv add opencv-python
```

## 使用例ファイル

`retinaface/examples.py` にさまざまな使用例が含まれています：

```bash
uv run python retinaface/examples.py
```

## ライセンス

MIT License

## 貢献

プルリクエストや Issue の報告をお待ちしています。

## 謝辞

このライブラリは以下のオリジナル実装をベースにしています：
- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
# RetinaFace 顔検出ライブラリ

RetinaFaceを使いやすいライブラリとして実装したPythonパッケージです。

## 特徴

- **簡単なAPI**: 数行のコードで顔検出が可能
- **高精度**: RetinaFaceアーキテクチャによる高精度な顔検出
- **ランドマーク検出**: 顔の特徴点（目、鼻、口角）も同時に検出
- **柔軟な入力**: 画像ファイル、numpy配列、動画ファイル、Webカメラに対応
- **複数モデル対応**: MobileNet 0.25（軽量・高速）とResNet50（高精度）を選択可能

## インストール

```bash
# 必要な依存関係をインストール
uv add torch torchvision opencv-python numpy

# 重みファイルをダウンロード（必要に応じて）
# ./weights/ ディレクトリに以下のファイルを配置:
# - mobilenet0.25.pth (MobileNet用)
# - resnet50.pth (ResNet50用)
```

## 基本的な使用方法

### 1. 基本的な顔検出

```python
from retinaface import RetinaFaceDetector

# 検出器を初期化
detector = RetinaFaceDetector(
    network='resnet50',  # または 'mobile0.25'
    confidence_threshold=0.8
)

# 顔検出を実行
faces = detector.detect('path/to/image.jpg', return_landmarks=True)

# 結果を表示
for i, face in enumerate(faces):
    print(f"顔 {i+1}:")
    print(f"  バウンディングボックス: {face['bbox']}")
    print(f"  信頼度: {face['confidence']:.4f}")
    if 'landmarks' in face:
        print(f"  ランドマーク: {face['landmarks']}")
```

### 2. 検出結果の可視化

```python
from retinaface import RetinaFaceDetector, load_image, draw_detection_results
import cv2

# 検出器を初期化
detector = RetinaFaceDetector(network='resnet50')

# 画像を読み込み
image = load_image('path/to/image.jpg')

# 顔検出
faces = detector.detect(image, return_landmarks=True)

# 結果を画像に描画
result_image = draw_detection_results(image, faces, show_landmarks=True)

# 結果を保存
cv2.imwrite('result.jpg', result_image)
```

### 3. 動画での顔検出

```python
detector = RetinaFaceDetector(network='mobile0.25')  # 高速化のため軽量モデルを使用

# 動画での顔検出
detector.detect_faces_in_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4',
    show_landmarks=True
)
```

### 4. Webカメラでのリアルタイム検出

```python
import cv2

detector = RetinaFaceDetector(network='mobile0.25')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 顔検出
    faces = detector.detect(frame, return_landmarks=True)
    
    # 結果を描画
    result_frame = detector.draw_detections(frame, faces)
    
    # 表示
    cv2.imshow('Face Detection', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## API リファレンス

### RetinaFaceDetector

メインの顔検出クラス

#### 初期化パラメータ

- `network` (str): 使用するネットワーク (`'mobile0.25'` または `'resnet50'`)
- `weights_path` (str, optional): 重みファイルのパス
- `device` (str): 使用するデバイス (`'auto'`, `'cpu'`, `'cuda'`)
- `confidence_threshold` (float): 信頼度の閾値 (デフォルト: 0.02)
- `nms_threshold` (float): NMSの閾値 (デフォルト: 0.4)
- `vis_threshold` (float): 可視化時の閾値 (デフォルト: 0.6)

#### メソッド

##### `detect(image, return_landmarks=True)`

顔検出を実行

**パラメータ:**
- `image`: 画像パス（str）またはnumpy配列
- `return_landmarks` (bool): ランドマークを返すかどうか

**戻り値:**
検出結果のリスト。各要素は以下のキーを持つ辞書:
- `'bbox'`: [x1, y1, x2, y2] バウンディングボックス
- `'confidence'`: 信頼度
- `'landmarks'`: (return_landmarks=Trueの場合) ランドマーク座標

##### `detect_faces_in_video(video_path, output_path=None, show_landmarks=True)`

動画内の顔検出を実行

##### `draw_detections(image, faces, show_landmarks=True)`

検出結果を画像に描画

### ユーティリティ関数

#### `load_image(image_path)`

画像を読み込み

#### `draw_detection_results(image, faces, show_landmarks=True, ...)`

検出結果を画像に描画（カスタマイズ可能）

#### `crop_face(image, bbox, margin_ratio=0.2)`

バウンディングボックスから顔領域を切り出し

#### `filter_faces_by_size(faces, min_area=100, max_area=None)`

顔のサイズでフィルタリング

#### `sort_faces_by_confidence(faces, descending=True)`

信頼度で顔をソート

## 設定例

### 高精度設定（ResNet50）

```python
detector = RetinaFaceDetector(
    network='resnet50',
    confidence_threshold=0.8,
    nms_threshold=0.4,
    vis_threshold=0.8
)
```

### 高速設定（MobileNet 0.25）

```python
detector = RetinaFaceDetector(
    network='mobile0.25',
    confidence_threshold=0.7,
    nms_threshold=0.4,
    vis_threshold=0.7
)
```

### CPU使用設定

```python
detector = RetinaFaceDetector(
    network='mobile0.25',
    device='cpu',
    confidence_threshold=0.8
)
```

## 性能比較

| モデル | 速度 | 精度 | 用途 |
|--------|------|------|------|
| MobileNet 0.25 | 高速 | 標準 | リアルタイム処理、動画処理 |
| ResNet50 | 標準 | 高精度 | 高精度が必要な用途、静止画処理 |

## トラブルシューティング

### 重みファイルが見つからない

```
FileNotFoundError: 重みファイルが見つかりません
```

→ `./weights/` ディレクトリに適切な重みファイルを配置してください。

### CUDA関連のエラー

```
RuntimeError: CUDA out of memory
```

→ CPUモードを使用するか、バッチサイズを小さくしてください：

```python
detector = RetinaFaceDetector(device='cpu')
```

### OpenCV関連のエラー

```
ImportError: No module named 'cv2'
```

→ OpenCVをインストールしてください：

```bash
uv add opencv-python
```

## 使用例ファイル

`retinaface/examples.py` にさまざまな使用例が含まれています：

```bash
uv run python retinaface/examples.py
```

## ライセンス

MIT License

## 貢献

プルリクエストや Issue の報告をお待ちしています。

## 謝辞

このライブラリは以下のオリジナル実装をベースにしています：
- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
