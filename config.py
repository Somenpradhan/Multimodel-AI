"""
config.py
Central configuration file for Multimodal AI System
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True


VISION_MODEL_PATH = "models/vision_model.pth"
AUDIO_MODEL_PATH = "models/audio_model.pth"
TEXT_MODEL_PATH = "models/text_model.pth"

YOLO_MODEL_PATH = "models/yolov8n.pt"


IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
AUDIO_DURATION = 3  # seconds


MAX_TEXT_LENGTH = 128
TOKENIZER_NAME = "distilbert-base-uncased"


VISION_WEIGHT = 0.4
AUDIO_WEIGHT = 0.3
TEXT_WEIGHT = 0.3

HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.4


RISK_LABELS = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

LOG_FILE = "logs/system.log"
SAVE_OUTPUTS = True