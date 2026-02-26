"""
vision_preprocess.py
High-performance vision preprocessing module
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
import config
import logging


class VisionPreprocessor:

    def __init__(self):
        self.device = config.DEVICE
        self.image_size = config.IMAGE_SIZE

        # Setup transform pipeline (fast + efficient)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD
            )
        ])

    def load_image(self, image_input):
        """
        Accepts:
        - file path (str)
        - numpy array (from webcam/OpenCV)
        """

        try:
            if isinstance(image_input, str):
                image = cv2.imread(image_input)

                if image is None:
                    raise ValueError(f"Unable to load image from path: {image_input}")

            elif isinstance(image_input, np.ndarray):
                image = image_input

            else:
                raise TypeError("Input must be file path or numpy array")

            return image

        except Exception as e:
            logging.error(f"[VisionPreprocessor] Image loading error: {e}")
            raise

    def preprocess(self, image_input):
        """
        Full preprocessing pipeline
        Returns tensor ready for model
        """

        try:
            # 1️⃣ Load image
            image = self.load_image(image_input)

            # 2️⃣ Convert BGR → RGB (OpenCV default fix)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 3️⃣ Apply transforms
            image_tensor = self.transform(image)

            # 4️⃣ Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            # 5️⃣ Move to device
            image_tensor = image_tensor.to(self.device)

            return image_tensor

        except Exception as e:
            logging.error(f"[VisionPreprocessor] Preprocessing failed: {e}")
            raise

    def preprocess_batch(self, image_list):
        """
        For multiple images (performance optimized)
        """

        try:
            tensors = []

            for img in image_list:
                tensor = self.preprocess(img)
                tensors.append(tensor)

            batch_tensor = torch.cat(tensors, dim=0)

            return batch_tensor

        except Exception as e:
            logging.error(f"[VisionPreprocessor] Batch preprocessing failed: {e}")
            raise