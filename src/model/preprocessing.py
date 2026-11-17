import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.models import MobileNet_V2_Weights


class ImagePreprocessor:
    """
    Prepares webcam frames before CNN inference.
    Processing:
    - BGR to RGB conversion
    - Image resizing and center cropping
    - Normalization using MobileNetV2 pretrained weights
    - Conversion to a PyTorch tensor
    """

    def __init__(self):
        self.weights = MobileNet_V2_Weights.DEFAULT
        self.transform = self.weights.transforms()

    def process(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV frame into a CNN-ready tensor.
        Args:
            frame: Image captured by OpenCV (BGR format).
        Returns:
            Tensor with shape (1, 3, 224, 224).
        """

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must have 3 color channels.")

        # 1. Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Convert NumPy array to PIL Image
        image = Image.fromarray(rgb_frame)

        # 3. Apply MobileNetV2 preprocessing
        tensor = self.transform(image)

        # 4. Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor