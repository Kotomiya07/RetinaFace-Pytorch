"""
RetinaFace使用例

このファイルでは、RetinaFaceライブラリの基本的な使用方法を示します。
"""

import os
import sys

# retinaface パッケージをインポート
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from retinaface import RetinaFaceDetector, load_image, draw_detection_results


def basic_face_detection_example():
    """基本的な顔検出の例"""
    print("=== 基本的な顔検出の例 ===")

    # 検出器を初期化
    detector = RetinaFaceDetector(
        network="resnet50",  # または 'mobile0.25'
        confidence_threshold=0.8,
        vis_threshold=0.6,
    )

    # 画像パス（存在する場合）
    image_path = "./test_image.jpg"

    if os.path.exists(image_path):
        # 顔検出を実行
        faces = detector.detect(image_path, return_landmarks=True)

        print(f"検出された顔の数: {len(faces)}")

        for i, face in enumerate(faces):
            print(f"  顔 {i + 1}:")
            print(f"    バウンディングボックス: {face['bbox']}")
            print(f"    信頼度: {face['confidence']:.4f}")
            if "landmarks" in face:
                print(f"    ランドマーク数: {len(face['landmarks'])}")

        # 結果を画像に描画
        image = load_image(image_path)
        result_image = draw_detection_results(image, faces, show_landmarks=True)

        # 結果を保存
        import cv2

        cv2.imwrite("detection_result.jpg", result_image)
        print("結果を detection_result.jpg に保存しました")
    else:
        print(f"テスト画像が見つかりません: {image_path}")


def batch_detection_example():
    """複数画像の一括処理の例"""
    print("\n=== 複数画像の一括処理の例 ===")

    # 検出器を初期化
    detector = RetinaFaceDetector(network="mobile0.25", confidence_threshold=0.7)

    # サンプル画像ディレクトリ
    image_dir = "./sample_images"

    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(image_dir, filename)

                try:
                    faces = detector.detect(image_path)
                    print(f"{filename}: {len(faces)}個の顔を検出")

                except Exception as e:
                    print(f"{filename}: エラー - {e}")
    else:
        print(f"サンプル画像ディレクトリが見つかりません: {image_dir}")


def video_detection_example():
    """動画での顔検出の例"""
    print("\n=== 動画での顔検出の例 ===")

    # 検出器を初期化
    detector = RetinaFaceDetector(network="resnet50", confidence_threshold=0.8, vis_threshold=0.7)

    # サンプル動画パス
    video_path = "./test_video.mp4"
    output_path = "./output_video.mp4"

    if os.path.exists(video_path):
        # 動画で顔検出を実行
        result_path = detector.detect_faces_in_video(video_path=video_path, output_path=output_path, show_landmarks=True)

        if result_path:
            print(f"結果動画を保存しました: {result_path}")
    else:
        print(f"テスト動画が見つかりません: {video_path}")


def webcam_detection_example():
    """Webカメラでのリアルタイム顔検出の例"""
    print("\n=== Webカメラでのリアルタイム顔検出の例 ===")

    try:
        import cv2

        # 検出器を初期化
        detector = RetinaFaceDetector(
            network="mobile0.25",  # リアルタイム処理のため軽量モデルを使用
            confidence_threshold=0.8,
            vis_threshold=0.8,
        )

        # Webカメラを開く
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Webカメラを開けませんでした")
            return

        print("Webカメラでの顔検出を開始します。'q'キーで終了します。")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 顔検出
            faces = detector.detect(frame, return_landmarks=True)

            # 結果を描画
            result_frame = detector.draw_detections(frame, faces, show_landmarks=True)

            # フレームを表示
            cv2.imshow("RetinaFace Detection", result_frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except ImportError:
        print("cv2がインストールされていません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def face_cropping_example():
    """顔の切り出し例"""
    print("\n=== 顔の切り出し例 ===")

    from retinaface.utils import crop_face

    # 検出器を初期化
    detector = RetinaFaceDetector(network="resnet50")

    image_path = "./test_image.jpg"

    if os.path.exists(image_path):
        # 画像を読み込み
        image = load_image(image_path)

        # 顔検出
        faces = detector.detect(image)

        print(f"検出された顔の数: {len(faces)}")

        # 各顔を切り出し
        for i, face in enumerate(faces):
            # 顔領域を切り出し
            face_image = crop_face(image, face["bbox"], margin_ratio=0.2)

            # 切り出した顔を保存
            import cv2

            output_path = f"face_{i + 1}.jpg"
            cv2.imwrite(output_path, face_image)
            print(f"顔 {i + 1} を {output_path} に保存しました")
    else:
        print(f"テスト画像が見つかりません: {image_path}")


def performance_comparison_example():
    """性能比較の例"""
    print("\n=== 性能比較の例 ===")

    import time

    image_path = "./test_image.jpg"

    if not os.path.exists(image_path):
        print(f"テスト画像が見つかりません: {image_path}")
        return

    # MobileNet 0.25での検出
    print("MobileNet 0.25での検出:")
    detector_mobile = RetinaFaceDetector(network="mobile0.25")

    start_time = time.time()
    faces_mobile = detector_mobile.detect(image_path)
    mobile_time = time.time() - start_time

    print(f"  検出時間: {mobile_time:.4f}秒")
    print(f"  検出数: {len(faces_mobile)}個")

    # ResNet50での検出
    print("ResNet50での検出:")
    detector_resnet = RetinaFaceDetector(network="resnet50")

    start_time = time.time()
    faces_resnet = detector_resnet.detect(image_path)
    resnet_time = time.time() - start_time

    print(f"  検出時間: {resnet_time:.4f}秒")
    print(f"  検出数: {len(faces_resnet)}個")

    print(f"速度差: {resnet_time / mobile_time:.2f}倍")


if __name__ == "__main__":
    print("RetinaFace ライブラリの使用例")
    print("=" * 50)

    # 基本的な使用例を実行
    basic_face_detection_example()

    # その他の例も実行可能
    # batch_detection_example()
    # video_detection_example()
    # webcam_detection_example()
    # face_cropping_example()
    # performance_comparison_example()

    print("\n使用例の実行完了!")
