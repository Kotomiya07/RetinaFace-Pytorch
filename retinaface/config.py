"""
RetinaFace設定ファイル
"""

# MobileNet 0.25設定
MOBILENET_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

# ResNet50設定
RESNET50_CONFIG = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": True,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}

# デフォルト検出パラメータ
DEFAULT_DETECTION_PARAMS = {
    "confidence_threshold": 0.02,
    "nms_threshold": 0.4,
    "vis_threshold": 0.6,
    "top_k": 5000,
    "keep_top_k": 750,
}

# サポートされているネットワーク
SUPPORTED_NETWORKS = ["mobile0.25", "resnet50"]

# デフォルト重みファイルパス
DEFAULT_WEIGHTS = {"mobile0.25": "./weights/mobilenet0.25.pth", "resnet50": "./weights/resnet50.pth"}
