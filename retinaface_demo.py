#!/usr/bin/env python3
"""
RetinaFace顔検出ライブラリの簡単なデモ

使用方法:
    uv run python retinaface_demo.py [画像パス]
"""

import argparse
import os
import sys
from pathlib import Path

# retinaface パッケージをインポート
from retinaface import RetinaFaceDetector, draw_detection_results, load_image


def main():
    parser = argparse.ArgumentParser(description="RetinaFace 顔検出デモ")
    parser.add_argument("image_path", nargs="?", default="examples/test_image.jpg", help="入力画像のパス (デフォルト: test_image.jpg)")
    parser.add_argument("--network", choices=["mobile0.25", "resnet50"], default="resnet50", help="使用するネットワーク")
    parser.add_argument("--confidence", type=float, default=0.8, help="信頼度の閾値")
    parser.add_argument("--output", default="result.jpg", help="出力画像のパス")
    parser.add_argument("--show-landmarks", action="store_true", help="ランドマークを表示")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="使用するデバイス")

    args = parser.parse_args()

    # 入力画像の存在確認
    if not os.path.exists(args.image_path):
        print(f"エラー: 画像ファイルが見つかりません: {args.image_path}")

        # test_image.jpgが存在する場合は提案
        if os.path.exists("test_image.jpg"):
            print("ヒント: test_image.jpg が見つかりました。以下のコマンドを試してください:")
            print(f"  uv run python {sys.argv[0]} test_image.jpg")

        return 1

    print(f"RetinaFace 顔検出デモ")
    print(f"入力画像: {args.image_path}")
    print(f"ネットワーク: {args.network}")
    print(f"信頼度閾値: {args.confidence}")
    print(f"デバイス: {args.device}")
    print("-" * 50)

    try:
        # 検出器を初期化
        print("検出器を初期化中...")
        detector = RetinaFaceDetector(
            network=args.network, confidence_threshold=args.confidence, vis_threshold=args.confidence, device=args.device
        )

        # 画像を読み込み
        print("画像を読み込み中...")
        image = load_image(args.image_path)
        print(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")

        # 顔検出を実行
        print("顔検出を実行中...")
        faces = detector.detect(image, return_landmarks=args.show_landmarks)

        print(f"検出された顔の数: {len(faces)}")

        # 検出結果を表示
        for i, face in enumerate(faces):
            print(f"  顔 {i + 1}:")
            print(f"    バウンディングボックス: {face['bbox']}")
            print(f"    信頼度: {face['confidence']:.4f}")

            if args.show_landmarks and "landmarks" in face:
                landmarks = face["landmarks"]
                print(f"    ランドマーク:")
                landmark_names = ["右目", "左目", "鼻", "右口角", "左口角"]
                for j, (x, y) in enumerate(landmarks):
                    name = landmark_names[j] if j < len(landmark_names) else f"点{j + 1}"
                    print(f"      {name}: ({x:.1f}, {y:.1f})")

        if len(faces) > 0:
            # 結果を画像に描画
            print("結果を描画中...")
            result_image = draw_detection_results(
                image, faces, show_landmarks=args.show_landmarks, vis_threshold=args.confidence
            )

            # 結果を保存
            import cv2

            cv2.imwrite(args.output, result_image)
            print(f"結果を保存しました: {args.output}")

            # 簡単な統計を表示
            confidences = [face["confidence"] for face in faces]
            areas = []
            for face in faces:
                bbox = face["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                areas.append(area)

            print("\n統計情報:")
            print(f"  平均信頼度: {sum(confidences) / len(confidences):.4f}")
            print(f"  最高信頼度: {max(confidences):.4f}")
            print(f"  最低信頼度: {min(confidences):.4f}")
            print(f"  平均顔サイズ: {sum(areas) / len(areas):.0f} pixels²")
            print(f"  最大顔サイズ: {max(areas):.0f} pixels²")
            print(f"  最小顔サイズ: {min(areas):.0f} pixels²")

        else:
            print("顔が検出されませんでした。")
            print("ヒント:")
            print("  - --confidence の値を下げてみてください (例: --confidence 0.5)")
            print("  - 別のネットワークを試してみてください (--network mobile0.25)")
            print("  - 画像に顔が写っているか確認してください")

        print("\n✓ 処理完了")
        return 0

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        if "weights" in str(e):
            print("ヒント: 重みファイルが見つかりません。")
            print("./weights/ ディレクトリに以下のファイルを配置してください:")
            print("  - mobilenet0.25.pth (MobileNet用)")
            print("  - resnet50.pth (ResNet50用)")
        return 1

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("\nデバッグ情報:")
        print(f"  Python version: {sys.version}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Image path exists: {os.path.exists(args.image_path)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
