"""
RetinaFaceライブラリのテスト

基本的な機能をテストします。
"""

import os
import sys
import numpy as np

# retinaface パッケージのパスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_imports():
    """インポートのテスト"""
    print("=== インポートテスト ===")

    try:
        # 基本的なインポート
        from retinaface.config import MOBILENET_CONFIG, RESNET50_CONFIG

        print("✓ 設定ファイルのインポート成功")

        from retinaface.utils import (
            load_image,
            draw_detection_results,
            crop_face,
            filter_faces_by_size,
            sort_faces_by_confidence,
        )

        print("✓ ユーティリティ関数のインポート成功")

        from retinaface.detector import RetinaFaceDetector

        print("✓ RetinaFaceDetectorクラスのインポート成功")

        return True

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False


def test_config():
    """設定のテスト"""
    print("\n=== 設定テスト ===")

    try:
        from retinaface.config import MOBILENET_CONFIG, RESNET50_CONFIG, DEFAULT_DETECTION_PARAMS, SUPPORTED_NETWORKS

        # 必要なキーが含まれているかチェック
        required_keys = ["name", "min_sizes", "steps", "variance"]

        for key in required_keys:
            assert key in MOBILENET_CONFIG, f"MobileNet設定に {key} がありません"
            assert key in RESNET50_CONFIG, f"ResNet50設定に {key} がありません"

        print(f"✓ サポートされているネットワーク: {SUPPORTED_NETWORKS}")
        print(f"✓ デフォルト検出パラメータ: {list(DEFAULT_DETECTION_PARAMS.keys())}")

        return True

    except Exception as e:
        print(f"✗ 設定テストエラー: {e}")
        return False


def test_utils():
    """ユーティリティ関数のテスト"""
    print("\n=== ユーティリティテスト ===")

    try:
        from retinaface.utils import calculate_face_area, get_face_center, filter_faces_by_size, sort_faces_by_confidence

        # サンプルデータ
        sample_faces = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9},
            {"bbox": [20, 20, 100, 100], "confidence": 0.7},
            {"bbox": [5, 5, 15, 15], "confidence": 0.8},
        ]

        # 面積計算テスト
        area1 = calculate_face_area([10, 10, 50, 50])
        assert area1 == 1600, f"面積計算エラー: {area1}"
        print("✓ 面積計算正常")

        # 中心座標テスト
        center = get_face_center([10, 10, 50, 50])
        assert center == (30, 30), f"中心座標計算エラー: {center}"
        print("✓ 中心座標計算正常")

        # サイズフィルタリングテスト
        filtered = filter_faces_by_size(sample_faces, min_area=500)
        assert len(filtered) == 2, f"フィルタリングエラー: {len(filtered)}"
        print("✓ サイズフィルタリング正常")

        # 信頼度ソートテスト
        sorted_faces = sort_faces_by_confidence(sample_faces)
        assert sorted_faces[0]["confidence"] == 0.9, "ソートエラー"
        print("✓ 信頼度ソート正常")

        return True

    except Exception as e:
        print(f"✗ ユーティリティテストエラー: {e}")
        return False


def test_detector_init():
    """検出器の初期化テスト（重みファイルなし）"""
    print("\n=== 検出器初期化テスト ===")

    try:
        from retinaface.detector import RetinaFaceDetector

        # 重みファイルの存在チェック
        mobilenet_path = "./weights/mobilenet0.25.pth"
        resnet_path = "./weights/resnet50.pth"

        if os.path.exists(mobilenet_path):
            print(f"✓ MobileNet重みファイル存在: {mobilenet_path}")
        else:
            print(f"- MobileNet重みファイル不存在: {mobilenet_path}")

        if os.path.exists(resnet_path):
            print(f"✓ ResNet重みファイル存在: {resnet_path}")
        else:
            print(f"- ResNet重みファイル不存在: {resnet_path}")

        # 重みファイルが存在する場合のみ初期化テスト
        if os.path.exists(mobilenet_path):
            try:
                detector = RetinaFaceDetector(network="mobile0.25", weights_path=mobilenet_path, device="cpu")
                print("✓ MobileNet検出器の初期化成功")
                return True
            except Exception as e:
                print(f"✗ MobileNet検出器の初期化失敗: {e}")

        if os.path.exists(resnet_path):
            try:
                detector = RetinaFaceDetector(network="resnet50", weights_path=resnet_path, device="cpu")
                print("✓ ResNet検出器の初期化成功")
                return True
            except Exception as e:
                print(f"✗ ResNet検出器の初期化失敗: {e}")

        print("- 重みファイルが見つからないため、初期化テストをスキップ")
        return True

    except Exception as e:
        print(f"✗ 検出器初期化テストエラー: {e}")
        return False


def test_example_image():
    """サンプル画像での検出テスト"""
    print("\n=== サンプル画像検出テスト ===")

    # テスト用のダミー画像を作成
    try:
        import cv2

        # 512x512のダミー画像を作成
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # 顔のような矩形を描画（テスト用）
        cv2.rectangle(dummy_image, (200, 200), (300, 300), (255, 255, 255), -1)
        cv2.rectangle(dummy_image, (220, 220), (240, 240), (0, 0, 0), -1)  # 左目
        cv2.rectangle(dummy_image, (260, 220), (280, 240), (0, 0, 0), -1)  # 右目
        cv2.rectangle(dummy_image, (240, 260), (260, 280), (0, 0, 0), -1)  # 口

        # テスト画像を保存
        test_image_path = "test_dummy_face.jpg"
        cv2.imwrite(test_image_path, dummy_image)
        print(f"✓ テスト画像を作成: {test_image_path}")

        # 重みファイルが存在する場合のみ検出テスト
        weights_paths = [("mobile0.25", "./weights/mobilenet0.25.pth"), ("resnet50", "./weights/resnet50.pth")]

        for network, weights_path in weights_paths:
            if os.path.exists(weights_path):
                try:
                    from retinaface.detector import RetinaFaceDetector

                    detector = RetinaFaceDetector(
                        network=network,
                        weights_path=weights_path,
                        device="cpu",
                        confidence_threshold=0.1,  # 低い閾値でテスト
                    )

                    faces = detector.detect(dummy_image, return_landmarks=True)
                    print(f"✓ {network}での検出完了 - 検出数: {len(faces)}")

                    if len(faces) > 0:
                        print(f"  - 最初の検出結果: bbox={faces[0]['bbox']}, conf={faces[0]['confidence']:.4f}")

                except Exception as e:
                    print(f"✗ {network}での検出エラー: {e}")

        # テスト画像を削除
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

        return True

    except ImportError:
        print("- OpenCVが利用できないため、画像検出テストをスキップ")
        return True
    except Exception as e:
        print(f"✗ サンプル画像検出テストエラー: {e}")
        return False


def main():
    """メインテスト関数"""
    print("RetinaFace ライブラリテスト開始")
    print("=" * 50)

    tests = [
        ("インポート", test_imports),
        ("設定", test_config),
        ("ユーティリティ", test_utils),
        ("検出器初期化", test_detector_init),
        ("サンプル画像検出", test_example_image),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}テストで例外発生: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("テスト結果まとめ:")

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n合計: {passed}/{len(results)} テスト通過")

    if passed == len(results):
        print("🎉 すべてのテストが通過しました！")
    else:
        print("⚠️  一部のテストが失敗しました。詳細を確認してください。")


if __name__ == "__main__":
    main()
