"""
RetinaFace顔検出器

使いやすいAPIを提供するRetinaFace顔検出クラス
"""

import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Union
from pathlib import Path

from .config import MOBILENET_CONFIG, RESNET50_CONFIG
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm


class RetinaFaceDetector:
    """
    RetinaFace顔検出器

    使用例:
        detector = RetinaFaceDetector(network='resnet50')
        faces = detector.detect('image.jpg')
    """

    def __init__(
        self,
        network: str = "resnet50",
        weights_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
        vis_threshold: float = 0.6,
    ):
        """
        RetinaFaceDetectorを初期化

        Args:
            network: 使用するネットワーク ('mobile0.25' or 'resnet50')
            weights_path: 重みファイルのパス (Noneの場合はデフォルトを使用)
            device: 使用するデバイス ('auto', 'cpu', 'cuda')
            confidence_threshold: 信頼度の閾値
            nms_threshold: NMSの閾値
            top_k: 上位K個を保持
            keep_top_k: NMS後に保持する上位K個
            vis_threshold: 可視化時の閾値
        """
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.vis_threshold = vis_threshold

        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 設定を取得
        if network == "mobile0.25":
            self.cfg = MOBILENET_CONFIG
        elif network == "resnet50":
            self.cfg = RESNET50_CONFIG
        else:
            raise ValueError(f"未対応のネットワーク: {network}")

        # 重みファイルのパスを設定
        if weights_path is None:
            if network == "mobile0.25":
                weights_path = "./weights/mobilenet0.25.pth"
            else:
                weights_path = "./weights/resnet50.pth"

        self.weights_path = weights_path

        # モデルを初期化
        self._load_model()

    def _load_model(self):
        """モデルを読み込み"""
        # モデルを作成
        self.net = RetinaFace(cfg=self.cfg, phase="test")

        # 重みを読み込み
        self.net = self._load_pretrained_model(self.net, self.weights_path)
        self.net.eval()

        # デバイスに移動
        self.net = self.net.to(self.device)

        # CuDNNを最適化
        if self.device.type == "cuda":
            cudnn.benchmark = True

        print(f"RetinaFace {self.network} モデルの読み込み完了!")

    def _load_pretrained_model(self, model, pretrained_path):
        """事前学習済みモデルを読み込み"""
        print(f"事前学習済みモデルを読み込み中: {pretrained_path}")

        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"重みファイルが見つかりません: {pretrained_path}")

        if self.device.type == "cpu":
            pretrained_dict = torch.load(pretrained_path, weights_only=False, map_location=lambda storage, loc: storage)
        else:
            device_id = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, weights_only=False, map_location=lambda storage, loc: storage.cuda(device_id))

        # state_dictキーがある場合は取り出す
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict["state_dict"], "module.")
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, "module.")

        # モデルに読み込み
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def _remove_prefix(self, state_dict, prefix):
        """state_dictからプレフィックスを削除"""
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def detect(self, image: Union[str, np.ndarray], return_landmarks: bool = True) -> List[dict]:
        """
        顔検出を実行

        Args:
            image: 画像パスまたはnumpy配列
            return_landmarks: ランドマークを返すかどうか

        Returns:
            検出結果のリスト。各要素は以下のキーを持つ辞書:
            - 'bbox': [x1, y1, x2, y2] バウンディングボックス
            - 'confidence': 信頼度
            - 'landmarks': (return_landmarks=Trueの場合) ランドマーク座標
        """
        # 画像を読み込み
        if isinstance(image, str):
            img_raw = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_raw is None:
                raise ValueError(f"画像を読み込めません: {image}")
        else:
            img_raw = image.copy()

        # 前処理
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # 推論
        with torch.no_grad():
            loc, conf, landms = self.net(img)

        # 後処理
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        # バウンディングボックスをデコード
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        # 信頼度を取得
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ランドマークをデコード
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # 低い信頼度を除外
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # 上位K個を保持
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # NMSを適用
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # 上位K個を保持（NMS後）
        dets = dets[: self.keep_top_k, :]
        landms = landms[: self.keep_top_k, :]

        # 結果を整理
        results = []
        for i, det in enumerate(dets):
            if det[4] < self.confidence_threshold:
                continue

            result = {"bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])], "confidence": float(det[4])}

            if return_landmarks and i < len(landms):
                landmarks = landms[i].reshape(5, 2)
                result["landmarks"] = landmarks.tolist()

            results.append(result)

        return results

    def detect_faces_in_video(
        self, video_path: str, output_path: Optional[str] = None, show_landmarks: bool = True
    ) -> Optional[str]:
        """
        動画内の顔検出を実行

        Args:
            video_path: 入力動画のパス
            output_path: 出力動画のパス (Noneの場合は表示のみ)
            show_landmarks: ランドマークを表示するかどうか

        Returns:
            出力動画のパス（出力した場合）
        """
        cap = cv2.VideoCapture(video_path)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 顔検出
                faces = self.detect(frame, return_landmarks=show_landmarks)

                # 検出結果を描画
                frame = self.draw_detections(frame, faces, show_landmarks)

                if output_path:
                    writer.write(frame)
                else:
                    cv2.imshow("Face Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if output_path:
                writer.release()
            cv2.destroyAllWindows()

        return output_path if output_path else None

    def draw_detections(self, image: np.ndarray, faces: List[dict], show_landmarks: bool = True) -> np.ndarray:
        """
        検出結果を画像に描画

        Args:
            image: 元画像
            faces: 検出結果
            show_landmarks: ランドマークを表示するかどうか

        Returns:
            描画後の画像
        """
        img_draw = image.copy()

        for face in faces:
            if face["confidence"] < self.vis_threshold:
                continue

            # バウンディングボックスを描画
            bbox = face["bbox"]
            cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            # 信頼度を表示
            text = f"{face['confidence']:.4f}"
            cv2.putText(img_draw, text, (bbox[0], bbox[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ランドマークを描画
            if show_landmarks and "landmarks" in face:
                landmarks = face["landmarks"]
                colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
                for i, (x, y) in enumerate(landmarks):
                    color = colors[i] if i < len(colors) else (255, 255, 255)
                    cv2.circle(img_draw, (int(x), int(y)), 1, color, 4)

        return img_draw
